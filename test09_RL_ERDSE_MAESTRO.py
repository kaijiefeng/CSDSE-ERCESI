import torch
import random
import numpy as np
import pdb
import time
import pandas
import os
from multiprocessing import Process, Lock, Manager, Pool
import sys

from config import config_global
sys.path.append("./util/")
from space import dimension_discrete, design_space, create_space_maestro
from actor import actor_policyfunction, get_log_prob, get_log_prob_rnn
from mlp import mlp_policyfunction, rnn_policyfunction
from evaluation_maestro import evaluation_maestro
from config_analyzer import config_self
from timer import timer
from recorder import recorder

class ERDSE():
	def __init__(self, iindex):
		self.iindex = iindex

		seed = self.iindex * 10000
		torch.manual_seed(seed)
		np.random.seed(seed)
		random.seed(seed)

		#### step1 assign model
		self.config = config_self(self.iindex)
		self.nnmodel = self.config.nnmodel
		self.constraints = self.config.constraints
		self.goal = self.config.goal
		self.target = self.config.target
		self.baseline = self.config.baseline
		self.config.config_check()
		self.pid = os.getpid()

		## initial DSE_action_space
		#self.layer_num = self.config.layer_num
		#self.DSE_action_space = create_space(self.layer_num)

		self.DSE_action_space = create_space_maestro(self.nnmodel, target = self.target)
		##initial evaluation
		self.evaluation = evaluation_maestro(self.iindex, self.nnmodel, self.pid, self.DSE_action_space)

		#define the hyperparameters
		self.PERIOD_BOUND = self.config.period
		self.SAMPLE_PERIOD_BOUND = 1
		self.GEMA = 0.999 #RL parameter, discount ratio
		self.ALPHA = 0.001 #RL parameter, learning step rate
		self.BATCH_SIZE = 1
		self.BASE_LINE = 0
		self.ENTROPY_RATIO = 0.1
		self.noise_std = 0.01

		#initial mlp_policyfunction, every action dimension owns a policyfunction
		#TODO:share the weight of first two layer among policyfunction
		action_scale_list = list()
		for dimension in self.DSE_action_space.dimension_box:
			action_scale_list.append(int(dimension.get_scale()))
		self.policy_type = "MLP"
		if(self.policy_type == "MLP"):
			#self.policyfunction = mlp_policyfunction(self.DSE_action_space.get_lenth(), action_scale_list)
			self.policyfunction = mlp_policyfunction(self.DSE_action_space.const_lenth + self.DSE_action_space.dynamic_lenth, action_scale_list)
		elif(self.policy_type == "RNN"):
			self.policyfunction = rnn_policyfunction(input_lenth=1, action_scale_list=action_scale_list)
		
		#print(f"model:{self.policyfunction.state_dict()}")
		#pdb.set_trace()

		##initial e_greedy_policy_function
		self.actor = actor_policyfunction()

		##initial optimizer
		self.policy_optimizer = torch.optim.Adam(
			self.policyfunction.parameters(), 
			lr=self.ALPHA, 
		)

		#### replay buffer, in order to record and reuse high return trace
		#### buffer is consist of trace list
		self.replay_buffer = list()

		#### data vision related
		self.best_objectvalue = 1000
		self.best_objectvalue_list = list()
		self.multiobjecvalue_list = list()
		self.t = timer()

	def train(self):
		self.t.start("all")
		period_bound = self.SAMPLE_PERIOD_BOUND + self.PERIOD_BOUND
		for period in range(period_bound):
			#print(f"period:{period}", end="\r")
			#here may need a initial function for action_space
			self.DSE_action_space.status_reset()
			rnn_state = None

			#store status, log_prob, reward and return
			status_list, action_list, return_list = list(), list(), list()
			reward_list = list()

			for step in range(self.DSE_action_space.get_lenth()): 
				#get status from S
				if(self.policy_type == "MLP"):
					current_status = self.DSE_action_space.get_compact_status(step)
				elif(self.policy_type == "RNN"):
					current_status = self.DSE_action_space.get_current_status(step)
				status_list.append(current_status)

				#use policy function to choose action and acquire log_prob
				#action, probs_noise = self.actor.action_choose_with_no_grad(self.policyfunction, self.DSE_action_space, step)
				if(self.policy_type == "MLP"):
					action = self.actor.action_choose_with_no_grad(self.policyfunction, self.DSE_action_space, current_status, step, std=self.noise_std)
				elif(self.policy_type == "RNN"):
					action, rnn_state = self.actor.action_choose_rnn(self.policyfunction, self.DSE_action_space, current_status, step, rnn_state, std=self.noise_std)
				action_list.append(action)

				#take action and get next state S'
				self.DSE_action_space.sample_one_dimension(step, action)

				#### in MC method, we can only sample in last step
				#### and compute reward R
		
				#TODO:design a good reward function
				if(step < (self.DSE_action_space.get_lenth() - 1)): #delay reward, only in last step the reward will be asigned
					reward = float(0)
				else:
					all_status = self.DSE_action_space.get_status()
					self.t.start("eva")
					metrics = self.evaluation.evaluate(all_status)
					self.t.end("eva")
					if(metrics != None):
						self.constraints.multi_update(metrics)
						objectvalue = metrics[self.goal] / self.baseline[self.goal]
						reward = 1 / (objectvalue * self.constraints.get_punishment())
					else:
						reward = 0

					#### recording
					if(period < self.SAMPLE_PERIOD_BOUND):
						pass
					else:
						if(objectvalue < self.best_objectvalue and self.constraints.is_all_meet()):
							self.best_objectvalue = objectvalue
							print(f"$$$$$iindex:{self.iindex}, best:{self.best_objectvalue}, metrics:{metrics}")
						self.best_objectvalue_list.append(self.best_objectvalue)
						self.multiobjecvalue_list.append([metrics["latency"], metrics["energy"]])
					#print(f"period:{period}, this:{objectvalue}, best:{self.best_objectvalue}, metrics:{metrics}, reward:{reward}", end = '\n')
					#print(f"period:{period}, iindex:{self.iindex}, action_list:{action_list[4:8]}, metrics:{metrics}, , baseline:{self.baseline[self.goal]}")
					#print(f"iindex:{self.iindex}, best:{self.best_objectvalue}, metrics:{metrics}")
				reward_list.append(reward)

			#compute and record return
			return_g = 0
			T = len(reward_list)
			for t in range(T):
				return_g = reward_list[T-1-t] + self.GEMA * return_g
				return_list.append(torch.tensor(return_g).reshape(1))
			return_list.reverse()

			#### record trace into buffer
			sample = {"reward":reward, "return_list":return_list, "status_list":status_list, "action_list":action_list, "obs":self.DSE_action_space.get_obs()}
			if(len(self.replay_buffer) < self.BATCH_SIZE):
				self.replay_buffer.append(sample)
			else:
				min_sample = min(self.replay_buffer, key = lambda sample:sample["reward"])
				if(sample["reward"] > min_sample["reward"]):
					index = self.replay_buffer.index(min_sample)
					self.replay_buffer[index] = sample
			
			if(period < self.SAMPLE_PERIOD_BOUND):
				pass
			elif(self.replay_buffer):
				#### compute loss and update actor network
				loss = torch.tensor(0)
				for _ in range(self.BATCH_SIZE):
					#### random sample trace from replay buffer
					sample_selected = random.choice(self.replay_buffer)
					s_return_list = sample_selected["return_list"]
					s_status_list = sample_selected["status_list"]
					s_action_list = sample_selected["action_list"]
					s_obs = sample_selected["obs"]

					#### compute log_prob and entropy
					T = self.DSE_action_space.get_lenth()
					sample_loss = torch.tensor(0)
					pi_noise = torch.tensor(1)
					if(self.policy_type == "MLP"):
						for t in range(T):
							s_entropy, s_log_prob = get_log_prob(self.policyfunction, self.DSE_action_space, s_status_list[t], s_action_list[t], t)
							return_item = -1 * s_log_prob * (s_return_list[t] - self.BASE_LINE)
							entropy_item = -1 * self.ENTROPY_RATIO * s_entropy
							sample_loss = sample_loss + return_item + entropy_item
					elif(self.policy_type == 'RNN'):
						rnn_state_train = None
						for t in range(T):
							s_entropy, s_log_prob, rnn_state_train = get_log_prob_rnn(self.policyfunction, s_obs[t], s_action_list[t], t, rnn_state_train)
							return_item = -1 * s_log_prob * s_return_list[t]
							entropy_item = -1 * self.ENTROPY_RATIO * s_entropy
							sample_loss = sample_loss + return_item + entropy_item
					#### accumulate loss
					sample_loss = sample_loss / T
					loss = loss + sample_loss
				loss = loss / self.BATCH_SIZE

				self.policy_optimizer.zero_grad()
				loss.backward()
				self.policy_optimizer.step()
			else:
				print("no avaiable sample")

			#self.timer.append({"Sampling:":(te_sampling - ts), "Evaluation":(te_evaluation - te_sampling), "Update":(te_update - te_evaluation)})
		#end for-period
		self.t.end("all")
	#end def-train

def run(args):
	iindex, objective_record, timecost_record, multiobjective_record = args
	print(f"%%%%TEST{iindex} START%%%%")
	DSE = ERDSE(iindex)
	DSE.train()
	timecost_list = DSE.t.get_list("all")
	evacost = DSE.t.get_sum("eva")
	timecost_list.append(evacost)
	
	DSE.best_objectvalue_list.append(iindex)
	timecost_list.append(iindex)
	DSE.multiobjecvalue_list.append([iindex, iindex])
	objective_record.append(DSE.best_objectvalue_list)
	timecost_record.append(timecost_list)
	multiobjective_record.append(DSE.multiobjecvalue_list)

if __name__ == '__main__':
	algoname = "ERDSE_MOO"
	use_multiprocess = True
	global_config = config_global()
	TEST_BOUND = global_config.TEST_BOUND
	PROCESS_NUM = global_config.PROCESS_NUM
	SCEN_TYPE = global_config.SCEN_TYPE
	SCEN_NUM = global_config.SCEN_NUM
	PASS = global_config.PASS

	args_list = list()
	objective_record = Manager().list()
	timecost_record = Manager().list()
	multiobjective_record = Manager().list()

	if(use_multiprocess):
		args_list = list()
		for iindex in range(TEST_BOUND):
			if(iindex in PASS): continue
			args_list.append((iindex, objective_record, timecost_record, multiobjective_record))
		pool = Pool(PROCESS_NUM)
		pool.map(run, args_list)
		pool.close()
		pool.join()
	else:
		for iindex in range(TEST_BOUND):
			if(iindex in PASS): continue
			run((iindex, objective_record, timecost_record))

	recorder(algoname, global_config, objective_record, timecost_record, multiobjective_record)



