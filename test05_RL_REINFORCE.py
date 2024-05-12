import torch
import random
import numpy as np
import pdb
import copy
import pandas
import os 
from multiprocessing import Process, Lock, Manager, Pool
import sys

from config import config_global
sys.path.append("./util/")
from space import dimension_discrete, design_space, create_space_maestro
from actor import actor_policyfunction
from mlp import mlp_policyfunction
from evaluation_maestro import evaluation_maestro
from config_analyzer import config_self
from timer import timer
from recorder import recorder

class RI():
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
		self.SAMPLE_PERIOD_BOUND = 1
		self.GEMA = 0.999 #RL parameter, discount ratio
		self.ALPHA = 0.001 #RL parameter, learning step rate
		self.THRESHOLD_RATIO = 2#0.05
		self.BATCH_SIZE = 1
		self.BASE_LINE = 0
		self.ENTROPY_RATIO = 0
		self.PERIOD_BOUND = self.config.period

		#initial mlp_policyfunction, every action dimension owns a policyfunction
		#TODO:share the weight of first two layer among policyfunction
		action_scale_list = list()
		for dimension in self.DSE_action_space.dimension_box:
			action_scale_list.append(int(dimension.get_scale()))
		#self.policyfunction = mlp_policyfunction(self.DSE_action_space.get_lenth(), action_scale_list)
		self.policyfunction = mlp_policyfunction(self.DSE_action_space.const_lenth + self.DSE_action_space.dynamic_lenth, action_scale_list)

		##initial e_greedy_policy_function
		self.actor = actor_policyfunction()

		##initial optimizer
		self.policy_optimizer = torch.optim.Adam(
			self.policyfunction.parameters(), 
			lr=self.ALPHA, 
		)

		#### loss replay buffer, in order to record and reuse high return trace
		self.loss_buffer = list()

		self.best_objectvalue = 1000
		self.best_objectvalue_list = list()
		self.multiobjecvalue_list = list()
		self.t = timer()

	def train(self):
		loss = torch.tensor(0)
		batch_index = 0

		self.t.start("all")
		for period in range(self.PERIOD_BOUND):
			#here may need a initial function for action_space
			self.DSE_action_space.status_reset()

			#store log_prob, reward and return
			entropy_list = list()
			log_prob_list = list()
			reward_list = list()
			return_list = list()

			for step in range(self.DSE_action_space.get_lenth()): 
				#get status from S
				current_status = self.DSE_action_space.get_compact_status(step)

				#use policy function to choose action and acquire log_prob
				entropy, action, log_prob_sampled = self.actor.action_choose(self.policyfunction, self.DSE_action_space, current_status, step)
				entropy_list.append(entropy)
				log_prob_list.append(log_prob_sampled)

				#take action and get next state S'
				self.DSE_action_space.sample_one_dimension(step, action)

				#### in MC method, we can only sample in last step
				#and compute reward R

				if(step < (self.DSE_action_space.get_lenth() - 1)): #delay reward, only in last step the reward will be asigned
					reward = float(0)
				else:
					#and compute reward R
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
					if(objectvalue < self.best_objectvalue and self.constraints.is_all_meet()):
						self.best_objectvalue = objectvalue
						print(f"period:{period}, this:{objectvalue}, best:{self.best_objectvalue}, reward:{reward}")
					self.best_objectvalue_list.append(self.best_objectvalue)
					self.multiobjecvalue_list.append([metrics["latency"], metrics["energy"]])

					#print(f"period:{period}, this:{objectvalue}, best:{self.best_objectvalue}, reward:{reward}", end = '\r')

				reward_list.append(reward)

			#compute and record return
			return_g = 0
			T = len(reward_list)
			for t in range(T):
				return_g = reward_list[T-1-t] + self.GEMA * return_g
				return_list.append(torch.tensor(return_g).reshape(1))
			return_list.reverse()
				
			#compute and record entropy_loss
			entropy_loss = torch.tensor(0)
			T = len(return_list)
			for t in range(T):
				retrun_item = -1 * log_prob_list[t] * (return_list[t] - self.BASE_LINE)
				entropy_item = -1 * self.ENTROPY_RATIO * entropy_list[t]
				entropy_loss = entropy_loss + retrun_item + entropy_item
			entropy_loss = entropy_loss / T

			loss = loss + entropy_loss
			batch_index = batch_index + 1

			#step update policyfunction
			if(period % self.BATCH_SIZE == 0):
				loss = loss / self.BATCH_SIZE
				self.policy_optimizer.zero_grad()
				loss.backward()
				self.policy_optimizer.step()

				loss = torch.tensor(0)
				batch_index = 0

			#end for-period
		#end def-train
		self.t.end("all")

	def test(self):
		for period in range(1):
			for step in range(self.DSE_action_space.get_lenth()):
				entropy, action, log_prob = self.actor.action_choose(self.policyfunction, self.DSE_action_space, step)
				self.fstatus = self.DSE_action_space.sample_one_dimension(step, action)
			self.evaluation.update_parameter(self.fstatus)
			self.fruntime, t_L = self.evaluation.runtime()
			self.fruntime = self.fruntime * 1000
			self.fpower = self.evaluation.power()
			print(
				"\n@@@@  TEST  @@@@\n",
				"final_status\n", self.fstatus, 
				"\nfinal_runtime\n", self.fruntime,
				"\npower\n", self.fpower,
				"\nbest_time\n", self.best_objectvalue_list[-1]
			)		

def run(args):
	iindex, objective_record, timecost_record, multiobjective_record = args
	print(f"%%%%TEST{iindex} START%%%%")
	DSE = RI(iindex)
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
	algoname = "RI_MOO"
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




