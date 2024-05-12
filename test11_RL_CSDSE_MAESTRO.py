import sys
import os
import torch
import random
import numpy as np
import pdb
import copy
import pandas
import multiprocessing
from multiprocessing import Process, Lock, current_process, Manager, Pool
from multiprocessing.managers import BaseManager

from config import config_global
sys.path.append("./dlrm")
from dlrm_tldse import dlrm_module
sys.path.append("./util/")
from space import dimension_discrete, design_space, create_space_maestro, tsne2D, tsne2D_fromfile
from actor import actor_random, actor_policyfunction, csdse_get_log_prob, get_log_prob_rnn
from mlp import mlp_policyfunction, rnn_policyfunction
from sample_buffer import buffer, warehouse
from evaluation_maestro import evaluation_maestro
from config_analyzer import config_self
from timer import timer
from recorder import recorder

class MyManager(BaseManager):
	pass
MyManager.register("buffer", buffer)
MyManager.register("warehouse", warehouse)

class NoDaemonProcess(multiprocessing.Process):
	def _get_daemon(self):
		return False
	def _set_daemon(self, value):
		pass
	daemon = property(_get_daemon, _set_daemon)

class Pool(multiprocessing.pool.Pool):
	Process = NoDaemonProcess

def compact(obs, design_space, dimension_index):
	compact_obs = []
	const_lenth = design_space.const_lenth
	dynamic_lenth = design_space.dynamic_lenth
	if(dimension_index < design_space.const_lenth):
		layer_index = 0
		layer = "Hardware"
		temp_layer = design_space.layer_name[layer_index]
	else:
		layer_index = int((dimension_index -const_lenth)/dynamic_lenth)
		layer = design_space.layer_name[layer_index]

	const_range = range(0, const_lenth)
	dynamic_range = range(const_lenth+layer_index*dynamic_lenth, const_lenth+(layer_index+1)*dynamic_lenth)
	if(layer == "Hardware"):
		for dindex in const_range:
			compact_obs.append(obs[dindex])
		for dindex in dynamic_range:
			compact_obs.append(0)
	else:
		for dindex in const_range:
			compact_obs.append(obs[dindex])
		for dindex in dynamic_range:	
			compact_obs.append(obs[dindex])
	compact_obs = np.array(compact_obs)	
	return compact_obs


class CSDSE():
	def __init__(self, iindex, rtype, jindex):
		self.iindex = iindex
		self.jindex = jindex
		self.rtype = rtype
		
		seed = self.iindex * 10000 + self.jindex
		
		print(f"rtype:{self.rtype}, seed:{seed}")
		torch.manual_seed(seed)
		np.random.seed(seed)
		random.seed(seed)
		os.environ['PYTHONHASHSEED'] = str(seed)

		#### step1 assign model
		self.config = config_self(self.iindex)
		self.nnmodel = self.config.nnmodel
		self.constraints = self.config.constraints
		self.goal = self.config.goal
		self.target = self.config.target
		self.baseline = self.config.baseline
		self.baseline_max = self.config.baseline_max
		if(rtype == "discreet_phd"):
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
		self.ENTROPY_RATIO = 0.1
		self.BATCH_SIZE = 1

		#define paramters of sample_buffer
		self.asample = dict()

		self.p_th = 50
		self.noise_std = 0.01
		self.is_compress = False
		self.patience = 0
		if(self.rtype == "brave_phd"):
			self.p_th = self.config.period
			self.is_compress = True
		elif(self.rtype == "discreet_phd"):
			self.p_th = self.config.period
			self.is_compress = False
		elif(self.rtype == "brave_master"):
			self.is_compress = True
		elif(self.rtype == "discreet_master"):
			self.is_compress = False
		elif(self.rtype == "brave_tutor"):
			self.is_compress = True
		elif(self.rtype == "discreet_tutor"):
			self.is_compress = False
		elif(self.rtype == "brave_reserve"):
			self.is_compress = True
		elif(self.rtype == "discreet_reserve"):
			self.is_compress = False
		else:
			print(f"type error: only support type 'phd', 'master', 'tutor', 'reserve'")

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
		#### fillter buffer
		self.has_fillter = False
		if(self.has_fillter):
			self.fillter_train_interval = 300
			self.fillter_obs_buffer = list()
			self.fillter_reward_buffer = list()
			den_idx, spa_idx, spa_shape = self.DSE_action_space.get_den_spa()
			self.dlrm = dlrm_module(m_out = 2, den_idx = den_idx, spa_idx = spa_idx, spa_shape = spa_shape)
			self.stack_record = 0

		#### data vision related
		self.best_status = dict()
		self.best_reward = 0
		self.best_objectvalue = 1000
		self.best_objectvalue_list = list()
		self.multiobjecvalue_list = list()
		self.t = timer()
		self.all_obs_list = list()
		self.all_reward_list = list()
		self.all_metric_list = list()
		
	def train(self, sample_buffer, sample_warehouse, lock):
		current_status = dict() #S
		next_status = dict() #S' 

		current_status_value = list()
		next_status_value = list()
		obs_list = list()

		self.initial_obs = self.DSE_action_space.get_obs() 
		self.initial_status = self.DSE_action_space.get_status()

		self.t.start("all")
		period = 0
		while(period < self.SAMPLE_PERIOD_BOUND + self.PERIOD_BOUND):
			#here may need a initial function for action_space
			self.DSE_action_space.status_reset()
			rnn_state = None

			#only brave_master will compress the design space
			if(self.is_compress):
				self.DSE_action_space.compress(self.nnmodel, period-self.SAMPLE_PERIOD_BOUND)
			else:
				pass

			#store reward
			reward_list = list()
			############################## sampling phase ################################
			self.t.start("sample")
			for step in range(self.DSE_action_space.get_lenth()): 
				#get status from S
				if(self.policy_type == "MLP"):
					current_status = self.DSE_action_space.get_compact_status(step)
				elif(self.policy_type == "RNN"):
					current_status = self.DSE_action_space.get_current_status(step)
				
				#use policy function to choose action and acquire log_prob
				if(self.policy_type == "MLP"):
					action = self.actor.action_choose_with_no_grad(self.policyfunction, self.DSE_action_space, current_status, step, self.noise_std, is_train = True)
				elif(self.policy_type == "RNN"):				
					action, rnn_state = self.actor.action_choose_rnn(self.policyfunction, self.DSE_action_space, current_status, step, rnn_state, std=self.noise_std)
				#if compressed, frozen parameters should be set to values from asample
				#unfrozen parameters will be set to the specific values from samples from policy
				if(self.is_compress and self.asample):
					self.DSE_action_space.set_one_dimension(step, self.asample["action_list"][step])
				#take action and get next state S'
				self.DSE_action_space.sample_one_dimension(step, action.item())

				if(step < (self.DSE_action_space.get_lenth() - 1)):
					reward = 0
					reward_list.append(reward)
			self.t.end("sample")

			#### action_list record, fillter repetitive status			
			action_list = self.DSE_action_space.get_action_list()

			if(self.has_fillter):
				if(period >= self.SAMPLE_PERIOD_BOUND):
					self.fillter_train_flag = ((period-self.SAMPLE_PERIOD_BOUND)%self.fillter_train_interval == 0) and ((period-self.SAMPLE_PERIOD_BOUND)!=0)
					if(self.fillter_train_flag):
						if(self.rtype == "discreet_phd"): 
							print(f"### {self.iindex}_{self.rtype}_start training") 
						#self.dlrm.train(self.fillter_obs_buffer, self.fillter_reward_buffer)
						if(self.rtype == "discreet_phd"): 
							print(f"   $$$ {self.iindex}_{self.rtype}_finish training")
				obs = self.DSE_action_space.get_obs_dlrm()
				predict = self.dlrm.predict(obs)
				
				if(predict[0][0] > 0.9 and not self.fillter_train_flag):
					self.stack_record += 1
					if(self.stack_record < 10): 
						continue
					else:
						self.stack_record = 0
						pass

			############################## sampling phase ################################

			#### in MC method, we can only sample in last step
			#### and compute the last reward R
			all_status = self.DSE_action_space.get_status()
			self.t.start("eva")
			metrics = self.evaluation.evaluate(all_status)
			self.t.end("eva")
			normalized_metrics = list()
			for key in metrics.keys():
				normalized_metrics.append(metrics[key]/self.baseline_max[key])
			if(metrics != None):
				self.constraints.multi_update(metrics)
				objectvalue = metrics[self.goal] / self.baseline[self.goal]
				reward = 1 / (objectvalue * self.constraints.get_punishment())
			else:
				reward = 0
			reward_list.append(reward)

			#### recording
			if(period < self.SAMPLE_PERIOD_BOUND):
				pass
			else:
				if(objectvalue < self.best_objectvalue and self.constraints.is_all_meet()):
					self.best_status = all_status
					self.best_objectvalue = objectvalue
					if(self.rtype == "discreet_phd"): print(f"****NEW**** period:{period}, iindex:{self.iindex}, pid:{self.pid}, rtype:{self.rtype}, best:{self.best_objectvalue}, metrics:{metrics}, buffer_lenth:{sample_buffer.size()}, threshold:{sample_buffer.threshold()}")
				self.best_objectvalue_list.append(self.best_objectvalue)
			self.multiobjecvalue_list.append([metrics["latency"], metrics["energy"]])
			if(self.rtype == "discreet_phd" and period % 100 == 0): print(f"period:{period}, iindex:{self.iindex}, pid:{self.pid}, rtype:{self.rtype}, best:{self.best_objectvalue}, metrics:{metrics}, buffer_lenth:{sample_buffer.size()}, threshold:{sample_buffer.threshold()}")
			#print(f"period:{period}, iindex:{self.iindex}, pid:{self.pid}, rtype:{self.rtype}, best:{self.best_objectvalue}, metrics:{metrics}, buffer_lenth:{sample_buffer.size()}, threshold:{sample_buffer.threshold()}")
			######################## information recording ########################

			#compute and record return
			return_list = list()
			return_g = 0
			T = len(reward_list)
			for t in range(T):
				return_g = reward_list[T-1-t] + self.GEMA * return_g
				return_list.append(return_g)
			return_list.reverse()

			if(self.has_fillter):
				#### record trace into buffer
				self.fillter_obs_buffer.append(self.DSE_action_space.get_obs_dlrm())
				self.fillter_reward_buffer.append(reward)
						 
			sample = dict()
			sample["obs"] = self.DSE_action_space.get_obs()
			sample["reward"] = reward
			sample["return_list"] = return_list
			sample["action_list"] = self.DSE_action_space.get_action_list()
			sample["metrics"] = metrics

			gsample = list(all_status.values()) + list(metrics.values())
			
			self.t.start("update")
			lock.acquire()
			sample_buffer.update(sample, self.jindex)
			sample_warehouse.update(gsample)
			lock.release()
			self.t.end("update")

			self.all_obs_list.append(sample["obs"])
			self.all_reward_list.append(sample["reward"])
			self.all_metric_list.append(normalized_metrics)

			######################### learning phase ##############################
			if(period < self.SAMPLE_PERIOD_BOUND):
				pass
			elif(sample_buffer):
				#### compute loss and update actor network
				self.t.start("train")
				loss = torch.tensor(0)
				entropy = torch.tensor(0)

				#### retrieve sample from shared buffer
				if(sample_buffer.size() == 0 or sample_buffer.is_empty()):
					print(f"there exist some unknown mistakes incuring the 0-buffer error")
					print(f"we catch the error but continue the process, the result may be not correct")
					print(f"following is error information:\n rtype:{self.rtype}, iindex:{self.iindex}, jindex:{self.jindex}")
					if(sample_buffer.size() == 0): print(f"!!sample_buffer size is 0")
					if(sample_buffer.is_empty()): print(f"!!sample_buffer is empty")
					self.asample = self.asample
				else:
					if(self.rtype == "brave_phd" or self.rtype == "discreet_phd"):
						self.asample = sample_buffer.acquire("max")
						self.best_reward = self.asample["reward"]
					elif(self.rtype == "brave_master" or self.rtype == "discreet_master"):
						if(self.patience == 0):
							self.asample = sample_buffer.acquire("better")
							self.patience += 1
						elif(self.patience <= self.p_th):
							self.patience += 1
						else:
							self.patience = 0
					elif(self.rtype == "brave_tutor" or self.rtype == "discreet_tutor"):
						if(self.patience == 0):
							self.asample = sample_buffer.acquire("random")
							self.patience += 1
						elif(self.patience <= self.p_th):
							self.patience += 1
						else:
							self.patience = 0
					elif(self.rtype == "brave_reserve" or self.rtype == "discreet_reserve"):
						stucking_ratio_best, stucking_ratio_better = sample_buffer.getstuckingstatus()
						if(stucking_ratio_best < 0.1): ## default: 0.1
							#### "best" learning strategy
							self.asample = sample_buffer.acquire("max")
							self.best_reward = self.asample["reward"]
						elif(stucking_ratio_better < 0.1): ## default: 0.1
							#### "better" learning strategy	
							if(self.patience == 0):
								self.asample = sample_buffer.acquire("better")
								self.patience += 1
							elif(self.patience <= self.p_th):
								self.patience += 1
							else:
								self.patience = 0
						else:
							#### "potential" learning strategy
							if(self.patience == 0):
								self.asample = sample_buffer.acquire("random")
								self.patience += 1
							elif(self.patience <= self.p_th):
								self.patience += 1
							else:
								self.patience = 0						
					else:
						print(f"State error: only support state max or state random")

				for _ in range(self.BATCH_SIZE):					
					s_obs = self.asample["obs"]
					s_return_list = self.asample["return_list"]
					s_action_list = self.asample["action_list"]

					#### compute log_prob and entropy
					T = self.DSE_action_space.get_lenth()
					sample_loss = torch.tensor(0)
					if(self.policy_type == "MLP"):
						for t in range(T):
							obs = copy.deepcopy(self.initial_obs)
							obs[0:t+1] = s_obs[0:t+1]
							compact_obs = compact(obs, self.DSE_action_space, t)
							action = s_action_list[t]
							return_g = torch.tensor(s_return_list[t]).reshape(1)

							s_entropy, s_log_prob = csdse_get_log_prob(self.policyfunction, compact_obs, action, t)
							retrun_item = -1 * s_log_prob * return_g
							sample_loss = sample_loss + retrun_item - 0.1 * s_entropy						
					elif(self.policy_type == "RNN"):
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
				self.t.end("train")
			else:
				print("no avaiable sample")
			######################### learning phase ##############################
			period = period + 1
		#end for-period
		self.t.end("all")
		#### save the dataflow and metrics files of the best sample
		is_print_bestresult = False
		if(is_print_bestresult): 
			if(self.best_status): 
				#print(f"This is {self.rtype}|{self.iindex}. Here we find the best design point:\n{self.best_status}")
				best_metric = self.evaluation.evaluate(self.best_status, save_files = True)
			else: print(f"This is {self.iindex}|{self.rtype}. There is no satisfied design point, please check the constraint defintions.")
	#end def-train

def task(args):
	iindex, member, jindex, temp_objective_record, temp_timecost_record, temp_multiobjective_record, sample_buffer, sample_warehouse, lock = args
	DSE = CSDSE(iindex, member, jindex)
	DSE.train(sample_buffer, sample_warehouse, lock)

	timecost_list = DSE.t.get_list("all")
	spcost = DSE.t.get_sum("sample")
	evacost = DSE.t.get_sum("eva")
	upcost = DSE.t.get_sum("update")
	trcost = DSE.t.get_sum("train")
	timecost_list.append(spcost)
	timecost_list.append(evacost)
	timecost_list.append(upcost)
	timecost_list.append(trcost)
	
	DSE.best_objectvalue_list.append(iindex)
	timecost_list.append(iindex)
	temp_objective_record.append(DSE.best_objectvalue_list)
	temp_timecost_record.append(timecost_list)
	temp_multiobjective_record.extend(DSE.multiobjecvalue_list)

	is_print_detail = False
	if(is_print_detail):
		detail_obj_path = "record/detail/obj/CSDSE_obj_{}_i={}_j={}_type={}.csv".format(DSE.goal, iindex, jindex, member)
		best_objectvalue_array = np.array(DSE.best_objectvalue_list).T
		best_objectvalue_df = pandas.DataFrame(best_objectvalue_array)
		best_objectvalue_df.to_csv(detail_obj_path, header = None, index = None)

		detail_obs_path = "record/detail/obs/CSDSE_obs_{}_i={}_j={}_type={}.csv".format(DSE.goal, iindex, jindex, member)
		obs_array = np.array(DSE.all_obs_list)
		obs_array_df = pandas.DataFrame(obs_array)
		obs_array_df.to_csv(detail_obs_path, header = None, index = None)

		detail_reward_path = "record/detail/reward/CSDSE_reward_{}_i={}_j={}_type={}.csv".format(DSE.goal, iindex, jindex, member)
		reward_array = np.array(DSE.all_reward_list)
		reward_array_df = pandas.DataFrame(reward_array)
		reward_array_df.to_csv(detail_reward_path, header = None, index = None)

		detail_metric_path = "record/detail/metric/CSDSE_metric_{}_i={}_j={}_type={}.csv".format(DSE.goal, iindex, jindex, member)
		metric_array = np.array(DSE.all_metric_list)
		metric_array_df = pandas.DataFrame(metric_array)
		metric_array_df.to_csv(detail_metric_path, header = None, index = None)

		#tsne2D(DSE.all_obs_list, DSE.all_reward_list, method="CSDSE_dis_{}_i={}_j={}_type={}".format(DSE.goal, iindex, jindex, member))

def run(args):
	iindex, objective_record, timecost_record, multiobjective_record = args
	print(f"%%%%TEST{iindex}(pid:{current_process().pid}) START%%%%")
	seed = iindex * 10000
	torch.manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)

	lock = Manager().Lock()

	config = config_self(iindex)
	design_space = create_space_maestro(config.nnmodel, config.target)
	metrics_name = config.metrics_name

	HRP, LRP, HRP_weight = design_space.corr_analysis("./data/corr_table_{}.csv".format(config.nnmodel))
	average_weight = [0 for i in range(design_space.get_lenth())]
	major_weight = [(index, 0.8 * weight) for index,weight in HRP_weight]
	minor_weight = 0.2 / len(LRP)
	for index, weight in major_weight:
		average_weight[index] = weight
	for index in LRP:
		average_weight[index] = minor_weight
	#print(f"weight:{average_weight}")

	mymanager = MyManager()
	mymanager.start()
	sample_warehouse = mymanager.warehouse(design_space, config, metrics_name)

	#member_list = ["discreet_phd", "discreet_phd", "discreet_master", "discreet_master", "discreet_tutor", "discreet_tutor", "discreet_reserve", "discreet_reserve"]
	#member_list = ["discreet_phd", "discreet_master", "discreet_tutor", "discreet_reserve", "discreet_reserve", "discreet_reserve", "discreet_reserve", "discreet_reserve"]
	member_list = ["brave_phd", "discreet_phd", "brave_master", "discreet_master", "brave_tutor", "discreet_tutor", "brave_reserve", "discreet_reserve"]
	#member_list = ["brave_phd"]*8 ## multi-agent ACDSE
	#member_list = ["discreet_phd"]*8 ## multi-agent ERDSE
	#member_list = ["discreet_master"]*8
	#member_list = ["discreet_tutor"]*8
	#member_list = ["discreet_reserve"]*8
	#member_list = ["discreet_phd","discreet_phd"]

	max_cnt = int(config.period*len(member_list))
	#sample_buffer = mymanager.buffer(distance_type = "Euclidean", distance_threshold = 0.4, adaptive = False, max_cnt = max_cnt, agent_cnt = len(member_list))
	#sample_buffer = mymanager.buffer(distance_type = "Weighted_Euclidean", distance_threshold = 0.3, weight = average_weight, adaptive = False, max_cnt = max_cnt, agent_cnt = len(member_list))
	#sample_buffer = mymanager.buffer(distance_type = "Euclidean", adaptive = True, max_cnt = max_cnt, agent_cnt = len(member_list))
	sample_buffer = mymanager.buffer(distance_type = "Weighted_Euclidean", weight = average_weight, adaptive = True, max_cnt = max_cnt, agent_cnt = len(member_list))
	
	if(os.path.exists("data/data_warehouse_{}.csv".format(config.nnmodel))):
		sample_warehouse.load("data/data_warehouse_{}.csv".format(config.nnmodel))
		sample_warehouse.load_baseline("data/baseline_{}.csv".format(config.nnmodel))
		#### Following code is implemented for sample reusing, which will be enable in future work
		#newbuffer = sample_warehouse.create_buffer()
		#sample_buffer.loadbuffer(newbuffer)

	jindex_list = [i for i in range(len(member_list))]
	args_list = list()
	temp_objective_record = Manager().list()
	temp_timecost_record = Manager().list()
	temp_obs_record = Manager().list()
	temp_multiobjective_record = Manager().list()

	for member, jindex in zip(member_list,jindex_list):
		args_list.append((iindex, member, jindex, temp_objective_record, temp_timecost_record, temp_multiobjective_record, sample_buffer, sample_warehouse, lock))
	pool = Pool(len(member_list))
	pool.map(task, args_list)
	pool.close()
	pool.join()
	#task(args_list[0])

	py_objective_record = list()
	py_objective_record = temp_objective_record[0:len(temp_objective_record)]
	py_objective_record = np.array(py_objective_record)
	py_objective_record = np.min(py_objective_record, axis = 0)
	objective_record.append(py_objective_record.tolist())

	py_timecost_record = list()
	py_timecost_record = temp_timecost_record[0:len(temp_objective_record)]
	py_timecost_record = np.array(py_timecost_record)
	py_timecost_record = np.mean(py_timecost_record, axis = 0)
	timecost_record.append(py_timecost_record.tolist())

	py_multiobjective_record = list()
	py_multiobjective_record = temp_multiobjective_record[0:len(temp_multiobjective_record)]
	py_multiobjective_record.append([iindex, iindex])
	multiobjective_record.append(py_multiobjective_record)

	is_print_buffer = True
	if(is_print_buffer):
		buffer_obs_path = "record/buffer/obs/CSDSE_obs_{}_{}_{}_i={}.csv".format(config.goal, config.nnmodel, config.target, iindex)
		buffer_reward_path = "record/buffer/reward/CSDSE_reward_{}_{}_{}_i={}.csv".format(config.goal, config.nnmodel, config.target, iindex)
		buffer_metrics_path = "record/buffer/metrics/CSDSE_metric_{}_{}_{}_i={}.csv".format(config.goal, config.nnmodel, config.target, iindex)

		buffer_obs_list, buffer_reward_list, buffer_metrics_list_neednorm = sample_buffer.getsampleinfo()

		buffer_obs_arary = np.array(buffer_obs_list)
		buffer_obs_df = pandas.DataFrame(buffer_obs_arary)
		buffer_obs_df.to_csv(buffer_obs_path, header = None, index = None)

		buffer_reward_array = np.array(buffer_reward_list)
		buffer_reward_df = pandas.DataFrame(buffer_reward_array)
		buffer_reward_df.to_csv(buffer_reward_path, header = None, index = None)

		buffer_metrics_list = list()
		for metrics in buffer_metrics_list_neednorm:
			normalized_metrics = list()
			for key in metrics.keys():
				normalized_metrics.append(metrics[key]/config.baseline_max[key])
			buffer_metrics_list.append(normalized_metrics)			
		buffer_metrics_array = np.array(buffer_metrics_list)
		buffer_metrics_df = pandas.DataFrame(buffer_metrics_array)
		buffer_metrics_df.to_csv(buffer_metrics_path, header = None, index = None)

	is_print_detail = False
	if(is_print_detail):
		obs_file_list = list()
		reward_file_list = list()
		for jindex, member in zip(jindex_list, member_list):
			obs_file_list.append("record/detail/obs/CSDSE_obs_{}_i={}_j={}_type={}.csv".format(config.goal, iindex, jindex, member))
			reward_file_list.append("record/detail/reward/CSDSE_reward_{}_i={}_j={}_type={}.csv".format(config.goal, iindex, jindex, member))
		has_interval = True
		interval = config.period

		tsne2D_fromfile(obs_file_list, reward_file_list, has_interval, interval)

if __name__ == '__main__':
	algoname = "CSDSE_HA_MS_WCB_MOO"
	use_multiprocess = True
	global_config = config_global()
	TEST_BOUND = global_config.TEST_BOUND
	#PROCESS_NUM = global_config.PROCESS_NUM
	PROCESS_NUM = 2
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
			#if(iindex != 1): continue
			run((iindex, objective_record, timecost_record, multiobjective_record))

	recorder(algoname, global_config, objective_record, timecost_record, multiobjective_record)




