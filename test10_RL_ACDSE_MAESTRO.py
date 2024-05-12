import torch
import random
import numpy as np
import pdb
import copy
import time
import pandas
import os
from multiprocessing import Process, Lock, Manager, Pool
import sys

from config import config_global
sys.path.append("./util/")
from space import dimension_discrete, design_space, create_space_maestro
from actor import actor_random, actor_policyfunction, get_log_prob, get_kldivloss_and_log_prob, get_kldivloss_and_log_prob_rnn
from mlp import mlp_policyfunction, mlp_fillter, rnn_policyfunction
from evaluation_maestro import evaluation_maestro
from config_analyzer import config_self
from timer import timer
from recorder import recorder

class ACDSE():
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

		self.DSE_action_space = create_space_maestro(self.nnmodel, target = self.target)
		##initial evaluation
		self.evaluation = evaluation_maestro(self.iindex, self.nnmodel, self.pid, self.DSE_action_space)

		#parameter mask list
		self.HRP, self.LRP, _  = self.DSE_action_space.corr_analysis("./data/corr_table_{}.csv".format(self.nnmodel))
		print(f"HRP:{self.HRP}")

		#define the hyperparameters
		self.PERIOD_BOUND = self.config.period
		self.SAMPLE_PERIOD_BOUND = 10#1000
		self.GEMA = 0.999 #RL parameter, discount ratio
		self.ALPHA = 0.001 #RL parameter, learning step rate
		self.BATCH_SIZE = 1
		self.KLDIV_RATIO = 1
		self.interval = 2
		self.noise_std = 0.01
		self.fillter_train_interval = 300

		self.alternative_search_ratio = 0.5
		self.fine_search_ratio = 0.1

		self.hungry_record = 0
		self.stack_record = 0
		self.decay_threshold = 50
		self.state_flag = None

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

		##initial e_greedy_policy_function
		self.actor = actor_policyfunction()

		##initial optimizer
		self.policy_optimizer = torch.optim.Adam(
			self.policyfunction.parameters(), 
			lr=self.ALPHA, 
		)

		##initial fillter
		#self.fillter = mlp_fillter(self.DSE_action_space.get_lenth())
		self.fillter = mlp_fillter(self.DSE_action_space.get_lenth())
		#### fillter buffer
		self.fillter_obs_buffer = list()
		self.fillter_reward_buffer = list()

		#### replay buffer, in order to record and reuse high return trace
		#### buffer is consist of trace list
		self.replay_buffer = list()

		#### data vision related
		self.best_objectvalue = 1000
		self.best_objectvalue_list = list()
		self.multiobjecvalue_list = list()
		self.t = timer()

	def more_local(self):
		self.KLDIV_RATIO = 1

	def more_global(self):
		self.KLDIV_RATIO = 0

	def train_fillter(self, obs_list, reward_list):
		print(f"**************  Training the fillter, now we have {len(obs_list)} samples   ******************")
		data_size = len(obs_list)
		batch_size = 50
		epoch_time = 1000

		refuse_record = 0
		reward_list_back = copy.deepcopy(reward_list)
		target_list = list()

		reward_list_back.sort()
		self.best_reward = max(reward_list)
		self.low_reward = reward_list_back[int(0.7*data_size)]
		print(f"***************   refuse_reward = {self.low_reward}, best_reward = {self.best_reward}     ******************")
		for reward in reward_list:
			if(reward <= self.low_reward): 
				target_list.append(0)
				refuse_record += 1
			else:
				target_list.append(1)
		#print(f"***************   refuse_ratio = {refuse_record/data_size}     ******************")

		temp_optimizer = torch.optim.Adam(self.fillter.parameters(), 0.001)
		loss_function = torch.nn.CrossEntropyLoss()
		data_all = np.array(obs_list)
		target_all = np.array(target_list)
	
		self.fillter.train()
		for epoch in range(epoch_time):
			idxs = np.random.randint(0, int(0.8*data_size), size=batch_size)
			data = torch.as_tensor(data_all[idxs], dtype=torch.float32)
			target = torch.as_tensor(target_all[idxs], dtype=torch.long)

			predict = self.fillter(data)
			loss = loss_function(predict, target)
			#if(epoch%200==0): print(f"loss:{loss}")

			temp_optimizer.zero_grad()
			loss.backward()
			temp_optimizer.step()
		##test

		self.fillter.eval()
		test_range = list(range(int(0.8 * data_size), data_size))
		data = torch.as_tensor(data_all[test_range], dtype=torch.float32)
		reward = np.array(reward_list)[test_range]
		predict = self.fillter(data)

		t_predict = list()
		for i in predict:
			if(i[0] > 0.9): t_predict.append(0)
			else: t_predict.append(1) 
		t_target = target_all[test_range]
		error = 0
		for i,j,d,r in zip(t_predict, t_target, data, reward):
			if(i != j): 
				error += 1
		print(f"************      accucy = {1 - error/len(test_range)}    ***************")
		#pdb.set_trace()

	def train(self):
		current_status = dict() #S
		next_status = dict() #S' 
		obs_list = list()

		self.t.start("all")
		period = 0
		while(period < self.SAMPLE_PERIOD_BOUND + self.PERIOD_BOUND):
			#here may need a initial function for action_space
			self.DSE_action_space.status_reset()
			rnn_state = None

			############################ adaptive compress #####################
			if(period < self.SAMPLE_PERIOD_BOUND):
				self.state_flag = "random"
				pass
			else:
				self.PMASK = range(0, self.DSE_action_space.get_lenth())
				HRP = self.HRP
				LRP = self.LRP
				if(self.constraints.is_any_margin_meet(self.fine_search_ratio)):
					self.state_flag = "fine"
					HRP = list(set(HRP).union(LRP))
					LRP = list(set(self.PMASK) ^ set(HRP))
					self.more_global()
					self.DSE_action_space.release_dimension(HRP)
					self.DSE_action_space.froze_dimension(LRP)
				elif(self.constraints.is_any_margin_meet(self.alternative_search_ratio)):
					if(self.hungry_record < self.decay_threshold):
						self.state_flag = "middle" 
						if((period - self.SAMPLE_PERIOD_BOUND) % self.interval != 0):
							self.DSE_action_space.release_dimension(HRP)
							self.DSE_action_space.froze_dimension(LRP)
						else:
							self.DSE_action_space.froze_dimension(HRP)
							self.DSE_action_space.release_dimension(LRP)
					else:
						self.state_flag = "middle to fine"
						HRP = list(set(HRP).union(LRP))
						LRP = list(set(self.PMASK) ^ set(HRP))
						self.more_global()
						self.DSE_action_space.release_dimension(HRP)
						self.DSE_action_space.froze_dimension(LRP)
				else:
					if(self.hungry_record < 2*self.decay_threshold):
						self.more_local()
						self.state_flag = "rough"
						self.DSE_action_space.release_dimension(HRP)
						self.DSE_action_space.froze_dimension(LRP)
					elif(self.hungry_record < self.decay_threshold):
						self.state_flag = "rough to middle" 
						if((period - self.SAMPLE_PERIOD_BOUND) % self.interval != 0):
							self.DSE_action_space.release_dimension(HRP)
							self.DSE_action_space.froze_dimension(LRP)
						else:
							self.DSE_action_space.froze_dimension(HRP)
							self.DSE_action_space.release_dimension(LRP)
					else:
						self.state_flag = "rough to fine"
						HRP = list(set(HRP).union(LRP))
						LRP = list(set(self.PMASK) ^ set(HRP))
						self.more_global()
						self.DSE_action_space.release_dimension(HRP)
						self.DSE_action_space.froze_dimension(LRP)
			############################ adaptive compress #####################
			
			#store status, log_prob, reward and return
			status_list = list()
			step_list = list()
			action_list = list()
			reward_list = list()
			return_list = list()

			############################## sampling phase ################################
			for step in range(self.DSE_action_space.get_lenth()): 
				step_list.append(step)

				#get status from S
				if(self.policy_type == "MLP"):
					current_status = self.DSE_action_space.get_compact_status(step)
				elif(self.policy_type == "RNN"):
					current_status = self.DSE_action_space.get_current_status(step)
				status_list.append(current_status)

				#use policy function to choose action and acquire log_prob
				if(self.policy_type == "MLP"):
					action = self.actor.action_choose_with_no_grad(self.policyfunction, self.DSE_action_space, current_status, step, self.noise_std, is_train = True)
				elif(self.policy_type == "RNN"):
					action, rnn_state = self.actor.action_choose_rnn(self.policyfunction, self.DSE_action_space, current_status, step, rnn_state, std=self.noise_std)

				###############################
				if(len(self.replay_buffer) < 1):
					pass
				else:
					sample_selected = random.choice(self.replay_buffer)
					self.DSE_action_space.set_one_dimension(step, sample_selected["action_list"][step])
				################################

				#take action and get next state S'
				self.DSE_action_space.sample_one_dimension(step, action)

				if(step < (self.DSE_action_space.get_lenth() - 1)):
					reward = 0
					reward_list.append(reward)
			############################## sampling phase ################################

			#### action_list record, fillter repetitive status			
			action_list = self.DSE_action_space.get_action_list()
			
			self.fillter_train_flag = (period-self.SAMPLE_PERIOD_BOUND)%self.fillter_train_interval == 0
			if(period >= self.SAMPLE_PERIOD_BOUND and self.fillter_train_flag):
				self.train_fillter(self.fillter_obs_buffer, self.fillter_reward_buffer)
			obs = self.DSE_action_space.get_obs()
			t_obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
			self.fillter.eval()
			predict = self.fillter(t_obs)
			if(predict[0][0] > 0.9 and not self.fillter_train_flag):
				self.stack_record += 1
				if(self.stack_record < 50): 
					continue
				else:
					self.stack_record = 0
					pass

			#### in MC method, we can only sample in last step
			#### and compute the last reward R

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
			reward_list.append(reward)
			print(f"index:{self.iindex}, period:{period}, state:{self.state_flag}, objectvalue:{objectvalue}, best:{self.best_objectvalue}", end="\r")
			#print(f"period:{period}, objectvalue:{objectvalue}, reward:{reward}", end = '\r')

			#### recording
			if(period < self.SAMPLE_PERIOD_BOUND):
				pass
			else:
				if(objectvalue < self.best_objectvalue and self.constraints.is_all_meet()):
					self.best_objectvalue = objectvalue
					self.constraints.update_stable_margin()
					#print(f"metrics:{metrics}")
					self.hungry_record = 0
					print(f"$$$$$iindex:{self.iindex}, best:{self.best_objectvalue}, metrics:{metrics}")
				else:
					self.hungry_record += 1
				self.best_objectvalue_list.append(self.best_objectvalue)
				self.multiobjecvalue_list.append([metrics["latency"], metrics["energy"]])

			if(not self.is_obsinlist(obs, self.fillter_obs_buffer)):
				self.fillter_obs_buffer.append(obs)
				self.fillter_reward_buffer.append(reward)

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

			######################### learning phase ##############################
			if(period < self.SAMPLE_PERIOD_BOUND):
				pass
			elif(len(self.replay_buffer) > 0):
				#### compute loss and update actor network
				loss = torch.tensor(0)
				entropy = torch.tensor(0)
				for _ in range(self.BATCH_SIZE):
					#### random sample trace from replay buffer
					sample_selected = random.choice(self.replay_buffer)
					s_return_list = sample_selected["return_list"]
					s_status_list = sample_selected["status_list"]
					s_action_list = sample_selected["action_list"]
					s_obs = sample_selected["obs"]

					#### compute log_prob and entropy
					T = len(s_return_list)
					sample_loss = torch.tensor(0)
					if(self.policy_type == "MLP"):
						for t in range(T):
							s_entropy, s_kldivloss, s_log_prob = get_kldivloss_and_log_prob( \
													             self.policyfunction,\
													             self.DSE_action_space, \
													             s_status_list[t], \
													             s_action_list[t], \
													             t
													             )
							return_item = -1 * s_log_prob * s_return_list[t]
							kldiv_item = s_kldivloss
							entropy_item = -1 * 0.1 * s_entropy
							sample_loss = sample_loss + (1 - self.KLDIV_RATIO) * (return_item) + self.KLDIV_RATIO * kldiv_item + entropy_item
					elif(self.policy_type == "RNN"):
						rnn_state_train = None
						for t in range(T):
							s_entropy, s_kldivloss, s_log_prob = get_kldivloss_and_log_prob_rnn( \
													             self.policyfunction,\
													             self.DSE_action_space, \
													             s_status_list[t], \
													             s_action_list[t], \
													             t,\
													             s_obs,\
													             rnn_state_train
													             )
							return_item = -1 * s_log_prob * s_return_list[t]
							kldiv_item = s_kldivloss
							entropy_item = -1 * 0.1 * s_entropy
							sample_loss = sample_loss + (1 - self.KLDIV_RATIO) * (return_item) + self.KLDIV_RATIO * kldiv_item + entropy_item
					#### accumulate loss
					sample_loss = sample_loss / T
					loss = loss + sample_loss
				loss = loss / self.BATCH_SIZE

				self.policy_optimizer.zero_grad()
				loss.backward()
				self.policy_optimizer.step()
			else:
				print("no avaiable sample")
			######################### learning phase ##############################

			period = period + 1
			#end for-period
		self.t.end("all")
		#end def-train

	def test(self):
		for period in range(1):
			for step in range(self.DSE_action_space.get_lenth()):
				action, _ = self.actor.action_choose_with_no_grad(self.policyfunction, self.DSE_action_space, step, is_train = False)
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
			
	def is_obsinlist(self, obs, obslist):
		for iobs in obslist:
			if(np.array_equal(obs, iobs)): return True
		return False	


def run(args):
	iindex, objective_record, timecost_record, multiobjective_record = args
	print(f"%%%%TEST{iindex} START%%%%")
	DSE = ACDSE(iindex)
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
	algoname = "ACDSE_MOO"
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




