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
from actor import actor_policyfunction, status_to_Variable, status_normalize, action_normalize
from replaybuffer import replaybuffer
from mlp import DDPG_mlp_qfunction as mlp_qfunction
from mlp import DDPG_mlp_policyfunction as mlp_policyfunction
from evaluation_maestro import evaluation_maestro
from config_analyzer import config_self
from timer import timer
from recorder import recorder

class DDPG():
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
		self.delay_reward = True
		self.WAIT_PERIOD = 1
		if(self.delay_reward): self.PERIOD_BOUND =  self.config.period
		else: self.PERIOD_BOUND = int(self.config.period/self.DSE_action_space.get_lenth())
		self.GEMA = 0.999 #RL parameter, discount ratio
		self.CRITIC_ALPHA = 0.01 #RL parameter, learning step rate
		self.ACTOR_ALPHA = 0.001 #RL parameter, learning step rate
		self.THRESHOLD_RATIO = 2#0.05 #RL parameter, reward punishment ratio
		self.TAU = 0.01 #RL parameter, target network update ratio
		self.BATCH_SIZE = 5 #RL parameter

		#compute action_scale_list and max_action_list
		self.action_scale_list = list()
		self.max_action_list = list()
		for dimension in self.DSE_action_space.dimension_box:
			self.action_scale_list.append(int(dimension.get_scale()))
			self.max_action_list.append(int(dimension.get_scale() - 1))

		##initial mlp_qfunction, which input is the vector of status
		'''
		self.critic = mlp_qfunction(self.DSE_action_space.get_lenth(), self.action_scale_list)
		self.target_critic = mlp_qfunction(self.DSE_action_space.get_lenth(), self.action_scale_list)
		'''
		self.critic = mlp_qfunction(self.DSE_action_space.const_lenth + self.DSE_action_space.dynamic_lenth, self.action_scale_list)
		self.target_critic = mlp_qfunction(self.DSE_action_space.const_lenth + self.DSE_action_space.dynamic_lenth, self.action_scale_list)
		self.target_critic.load_state_dict(self.critic.state_dict())

		#initial mlp_policyfunction, every action dimension owns a policyfunction
		'''
		self.actor = mlp_policyfunction(self.DSE_action_space.get_lenth(), self.action_scale_list)
		self.target_actor = mlp_policyfunction(self.DSE_action_space.get_lenth(), self.action_scale_list)
		'''
		self.actor = mlp_policyfunction(self.DSE_action_space.const_lenth + self.DSE_action_space.dynamic_lenth, self.action_scale_list)
		self.target_actor = mlp_policyfunction(self.DSE_action_space.const_lenth + self.DSE_action_space.dynamic_lenth, self.action_scale_list)
		self.target_actor.load_state_dict(self.actor.state_dict())

		##initial e_greedy_policy_function
		self.action_choose = actor_policyfunction()

		##initial optimizer
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr = self.CRITIC_ALPHA)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = self.ACTOR_ALPHA)

		##initial replaybuffer
		self.replaybuffer = replaybuffer()

		self.best_objectvalue = 1000
		self.best_objectvalue_list = list()
		self.t = timer()

	def train(self):
		current_status = dict() #S
		next_status = dict() #S' 

		self.t.start("all")
		for period in range(self.WAIT_PERIOD + self.PERIOD_BOUND):
			#here may need a initial function for action_space
			self.DSE_action_space.status_reset()

			#store reward and return
			reward_list = list()
			return_list = list()
	
			for step in range(self.DSE_action_space.get_lenth()): 
				#get status from S
				current_status = self.DSE_action_space.get_compact_status(step)

				#use policy function to choose action and acquire log_prob
				action_index, action_tensor = self.action_choose.action_choose_DDPG(self.actor, self.DSE_action_space, current_status, step)

				#take action and get next state S'
				self.DSE_action_space.sample_one_dimension(step, action_index)
				next_status = self.DSE_action_space.get_compact_status(step)

				if(step < (self.DSE_action_space.get_lenth() - 1)): #delay reward, only in last step the reward will be asigned
					not_done = 1
					if(self.delay_reward):
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

						print(f"period:{period}, objectvalue:{objectvalue}, reward:{reward}", end = '\r')

						#### recording
						if(period < self.WAIT_PERIOD):
							pass
						else:
							if(objectvalue < self.best_objectvalue and self.constraints.is_all_meet()):
								self.best_objectvalue = objectvalue
							self.best_objectvalue_list.append(self.best_objectvalue)						
				else:
					not_done = 0
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

					print(f"period:{period}, objectvalue:{objectvalue}, reward:{reward}", end = '\r')

					#### recording
					if(period < self.WAIT_PERIOD):
						pass
					else:
						if(objectvalue < self.best_objectvalue and self.constraints.is_all_meet()):
							self.best_objectvalue = objectvalue
						self.best_objectvalue_list.append(self.best_objectvalue)

				self.replaybuffer.add(current_status, action_tensor, next_status, reward, step, not_done)

				#assign next_status to current_status
				current_status = next_status

				#####   START TO TRAIN    ######
				if(period < self.WAIT_PERIOD):
					pass
				else:
					critic_loss_tensor = torch.tensor(0)
					actor_loss_tensor = torch.tensor(0)
					rp_current_status_list = list()
					rp_action_list = list()
					rp_next_status_list = list()
					rp_reward_list = list()
					rp_step_list = list()
					rp_not_done_list = list()

					for index in range(self.BATCH_SIZE):
						sp_current_status, sp_action, sp_next_status, sp_reward, sp_step, sp_not_done = copy.deepcopy(self.replaybuffer.sample())
						rp_current_status_list.append(sp_current_status)
						rp_action_list.append(sp_action)
						rp_next_status_list.append(sp_next_status)
						rp_reward_list.append(sp_reward)
						rp_step_list.append(sp_step)
						rp_not_done_list.append(sp_not_done)

					for sp_current_status, sp_action, sp_next_status, sp_reward, sp_step, sp_not_done in zip(rp_current_status_list, rp_action_list, rp_next_status_list, rp_reward_list, rp_step_list, rp_not_done_list):
						sp_next_step = (sp_step + 1) % self.DSE_action_space.get_lenth()

						#normalize status
						status_normalize(sp_current_status, self.DSE_action_space)
						status_normalize(sp_next_status, self.DSE_action_space)
						current_status = status_to_Variable(sp_current_status)
						next_status = status_to_Variable(sp_next_status)

						#normalize action_tensor
						action = sp_action
						next_action = self.target_actor(next_status, sp_next_step)
						#action_normalize(action, self.DSE_action_space, sp_step)
						#action_normalize(next_action, self.DSE_action_space, sp_next_step)
						
						#compute target q value
						target_q = self.target_critic(
							next_status, 
							next_action,
							sp_next_step
						)
						target_q = sp_reward + (sp_not_done * self.GEMA * target_q).detach()

						#compute current q value
						current_q = self.critic(
							current_status,
							action,
							sp_step
						)

						#print(f"current_q:{current_q}, target_q:{target_q}, reward:{sp_reward}")
						#compute critic_loss
						critic_loss = torch.nn.functional.mse_loss(current_q, target_q)
						critic_loss_tensor = critic_loss_tensor + critic_loss
					critic_loss_tensor = critic_loss_tensor / self.BATCH_SIZE
					#update critic
					self.critic_optimizer.zero_grad()
					critic_loss_tensor.backward()
					self.critic_optimizer.step()

					for sp_current_status, sp_action, sp_next_status, sp_reward, sp_step, sp_not_done in zip(rp_current_status_list, rp_action_list, rp_next_status_list, rp_reward_list, rp_step_list, rp_not_done_list):
						#normalize status
						status_normalize(sp_current_status, self.DSE_action_space)
						status_normalize(sp_next_status, self.DSE_action_space)
						current_status = status_to_Variable(sp_current_status)
						next_status = status_to_Variable(sp_next_status)		

						#normalize action_tensor
						action = self.actor(current_status, sp_step)	
						#action_normalize(action, self.DSE_action_space, sp_step)	

						#compute actor_loss
						actor_loss = -self.critic(
							current_status,
							action,
							sp_step
						)
						actor_loss_tensor = actor_loss_tensor + actor_loss
					actor_loss_tensor = actor_loss_tensor / self.BATCH_SIZE
					#update actor
					self.actor_optimizer.zero_grad()
					actor_loss_tensor.backward()
					self.actor_optimizer.step()

					# Update the frozen target models
					for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
						target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

					for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
						target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)
				
				
				#end for step
			#end for period
		self.t.end("all")
		#end train()

	def test(self):
		#### create proxy agent in order to protect status
		proxy_space = copy.deepcopy(self.DSE_action_space)
		proxy_evaluation = copy.deepcopy(self.evaluation)
		for period in range(1):
			for step in range(proxy_space.get_lenth()):
				action, _ = self.action_choose.action_choose_DDPG(self.actor, proxy_space, step)
				self.fstatus = proxy_space.sample_one_dimension(step, action)
			proxy_evaluation.update_parameter(self.fstatus)
			self.fruntime, t_L = proxy_evaluation.runtime()
			self.fpower = proxy_evaluation.power()
			self.fruntime = self.fruntime * 1000
			print(
				"@@@@  TEST  @@@@\n",
				"final_status\n", self.fstatus, 
				"\nfinal_runtime\n", self.fruntime,
				"\nfinal_power\n", self.fpower,
				"\nbest_runtime\n", self.best_runtime_list[-1],
			)	


def run(args):
	iindex, objective_record, timecost_record = args
	print(f"%%%%TEST{iindex} START%%%%")
	DSE = DDPG(iindex)
	DSE.train()

	timecost_list = DSE.t.get_list("all")
	evacost = DSE.t.get_sum("eva")
	timecost_list.append(evacost)

	DSE.best_objectvalue_list.append(iindex)
	timecost_list.append(iindex)
	objective_record.append(DSE.best_objectvalue_list)
	timecost_record.append(timecost_list)

if __name__ == '__main__':
	algoname = "DDPG"
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

	if(use_multiprocess):
		args_list = list()
		for iindex in range(TEST_BOUND):
			if(iindex in PASS): continue
			args_list.append((iindex, objective_record, timecost_record))
		pool = Pool(PROCESS_NUM)
		pool.map(run, args_list)
		pool.close()
		pool.join()
	else:
		for iindex in range(TEST_BOUND):
			if(iindex in PASS): continue
			run((iindex, objective_record, timecost_record))

	recorder(algoname, global_config, objective_record, timecost_record)



