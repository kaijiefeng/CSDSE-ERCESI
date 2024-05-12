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
from actor import actor_e_greedy, status_to_Variable, status_normalize
from replaybuffer import replaybuffer
from mlp import mlp_qfunction
from evaluation_maestro import evaluation_maestro
from config_analyzer import config_self
from timer import timer
from recorder import recorder

class DQN():
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

		#hyper parameters
		self.delay_reward = True
		if(self.delay_reward): self.PERIOD_BOUND = self.config.period
		else: self.PERIOD_BOUND = int(self.config.period/self.DSE_action_space.get_lenth())
		self.GEMA = 0.999 #RL parameter, discount ratio, GEMA = 0 means we use reward as return
		self.ALPHA = 0.01 #RL parameter, learning step rate
		self.WAIT_PERIOD = 1 #RL parameter, for fill replaybuffer
		self.BATCH_SIZE = 5 #RL parameter, as its name say
		self.TAU = 0.01
		
		##initial mlp_qfunction, which input is the vector of status
		'''
		self.qfunction = mlp_qfunction(self.DSE_action_space.get_lenth())
		self.target_qfunction = mlp_qfunction(self.DSE_action_space.get_lenth())
		'''
		self.qfunction = mlp_qfunction(self.DSE_action_space.const_lenth + self.DSE_action_space.dynamic_lenth)
		self.target_qfunction = mlp_qfunction(self.DSE_action_space.const_lenth + self.DSE_action_space.dynamic_lenth)
		self.target_qfunction.load_state_dict(self.qfunction.state_dict())

		##initial e_greedy_policy_function
		self.actor = actor_e_greedy()

		##initial replaybuffer
		self.replaybuffer = replaybuffer()

		#initial optimizer
		self.optimizer = torch.optim.Adam(
			self.qfunction.parameters(), 
			lr=self.ALPHA, 
		)

		self.best_objectvalue = 1000
		self.best_objectvalue_list = list()
		self.t = timer()

	def train(self):
		current_status = dict() #S
		next_status = dict() #S' 
		loss_batch = torch.tensor(0)
		loss_index = 0

		self.t.start("all")
		for period in range(self.WAIT_PERIOD + self.PERIOD_BOUND):
			#### here may need a initial function for action_space
			ratio = 1 - period/self.PERIOD_BOUND

			self.DSE_action_space.status_reset()
			for step in range(self.DSE_action_space.get_lenth()): 
				#### get status from S
				current_status = self.DSE_action_space.get_compact_status(step)
				#### use e-greedy algorithm get action A, action is the index of best action
				#### belong to that dimension
				action = self.actor.action_choose(self.qfunction, self.DSE_action_space, step, ratio)
				#### take action and get next state S'
				self.DSE_action_space.sample_one_dimension(step, action)
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

					print(f"period:{period}, objectvalue:{objectvalue}, reward:{reward}", end = '\r')

					#### recording
					if(period < self.WAIT_PERIOD):
						pass
					else:
						if(objectvalue < self.best_objectvalue and self.constraints.is_all_meet()):
							self.best_objectvalue = objectvalue
						self.best_objectvalue_list.append(self.best_objectvalue)

				#### push record to replaybuffer
				self.replaybuffer.add(current_status, action, next_status, reward, step, not_done)

				#### assign next_status to current_status
				current_status = next_status

				#################################START TRAIN###########################################
				#### batch train

				if(period < self.WAIT_PERIOD):
					pass
				else:
					for index in range(self.BATCH_SIZE):
						#### randomly sample from replaybuffer
						sp_current_status, sp_action, sp_next_status, sp_reward, sp_step, sp_not_done = copy.deepcopy(self.replaybuffer.sample())

						#### compute v(S',w)
						if(sp_not_done):
							sp_next_status = status_normalize(sp_next_status, self.DSE_action_space)
							next_variable = status_to_Variable(sp_next_status)
							next_cnt = (sp_step + 1) / self.DSE_action_space.get_lenth()
							next_cnt = torch.tensor(next_cnt).float().view(1)
							next_variable = torch.cat((next_variable, next_cnt), dim=-1)
							next_qvalue = self.target_qfunction(next_variable)

						#### compute return g
						if(sp_not_done):
							return_g = torch.tensor(sp_reward).float() + self.GEMA * next_qvalue.detach()
						else:
							return_g = torch.tensor(sp_reward).float()
						#### compute v(S,w)
						sp_current_status = status_normalize(sp_current_status, self.DSE_action_space)
						current_variable = status_to_Variable(sp_current_status)
						current_cnt = sp_step / self.DSE_action_space.get_lenth()
						current_cnt = torch.tensor(current_cnt).float().view(1)
						current_variable = torch.cat((current_variable, current_cnt), dim=-1)
						current_qvalue = self.qfunction(current_variable) 

						#### train the qfunction
						loss = torch.nn.functional.mse_loss(current_qvalue, return_g)

						#### accumalate loss
						loss_batch = loss_batch + loss
						loss_index = loss_index + 1

						#### end for self.BATCH_SIZE

					#### update network
					loss_batch = loss_batch / loss_index
					self.optimizer.zero_grad()
					loss_batch.backward()
					self.optimizer.step()

					loss_batch = torch.tensor(0)
					loss_index = 0

					#### gradually update target network
					for param, target_param in zip(self.qfunction.parameters(), self.target_qfunction.parameters()):
						target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)
				#### end of for step
			#### end of for period
		self.t.end("all")

	#### end of def train
	def test(self):
		#### create proxy agent in order to protect status
		proxy_space = copy.deepcopy(self.DSE_action_space)
		proxy_evaluation = copy.deepcopy(self.evaluation)
		for period in range(1):
			for step in range(proxy_space.get_lenth()):
				action = self.actor.best_action_choose(self.qfunction, proxy_space, step)
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

	def qfunction_check(self):
		proxy_space = copy.deepcopy(self.DSE_action_space)
		proxy_evaluation = copy.deepcopy(self.evaluation)
		
		with torch.no_grad():
			for index in range(1000):
				current_status, action, next_status, reward, step, not_done = copy.deepcopy(self.replaybuffer.sample())
				
				if(not_done):
					next_status = status_normalize(next_status, proxy_space)
					next_variable = status_to_Variable(next_status)
					next_cnt = (step+1) / proxy_space.get_lenth()
					next_cnt = torch.tensor(next_cnt).float().view(1)
					next_variable = torch.cat((next_variable, next_cnt), dim = -1)
					next_qvalue = self.target_qfunction(next_variable)

					return_g = torch.tensor(reward).float() + self.GEMA * next_qvalue.detach()
				else:
					return_g = torch.tensor(reward).float()
				
				current_status = status_normalize(current_status, proxy_space)
				current_variable = status_to_Variable(current_status)
				current_cnt = step / proxy_space.get_lenth()
				current_cnt = torch.tensor(current_cnt).float().view(1)
				current_variable = torch.cat((current_variable, current_cnt), dim = -1)
				current_qvalue = self.qfunction(current_variable) 

				self.worksheet.write(index+1, 0, return_g.item())
				self.worksheet.write(index+1, 1, current_qvalue.item())

			self.workbook.save("record/reward&return/qfunction-TD method.xls")

def run(args):
	iindex, objective_record, timecost_record = args
	print(f"%%%%TEST{iindex} START%%%%")
	DSE = DQN(iindex)
	DSE.train()

	timecost_list = DSE.t.get_list("all")
	evacost = DSE.t.get_sum("eva")
	timecost_list.append(evacost)

	DSE.best_objectvalue_list.append(iindex)
	timecost_list.append(iindex)
	objective_record.append(DSE.best_objectvalue_list)
	timecost_record.append(timecost_list)

if __name__ == '__main__':
	algoname = "DQN"
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


    

