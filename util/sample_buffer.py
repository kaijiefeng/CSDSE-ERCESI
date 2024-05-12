import copy
import random
import torch
import pandas
import pdb
import numpy
import os 

class buffer():
	def __init__(self, distance_type = "Euclidean", distance_threshold = 0, weight = [], adaptive = False, max_cnt=8000, agent_cnt = 8):
		self.sample_buffer = list()
		self.distance_type = distance_type
		self.distance_threshold = distance_threshold
		self.best_reward = 0
		self.better_reward = 0
		self.stucking_cnt_best = 0
		self.stucking_ratio_best = 0
		self.stucking_cnt_better = 0
		self.stucking_ratio_better = 0
		self.sample_cnt = 0
		self.is_update = True
		self.weight = weight

		self.adaptive = adaptive
		#### sampling_window_lenth defines the sampling range of m&p agents
		self.sampling_window_lenth = 10
		self.agent_cnt = agent_cnt
		self.expect_size_lb = int(2*self.agent_cnt)
		self.expect_size_ub = int(3*self.agent_cnt)
		self.threshold_step = 0.01	

		self.max_cnt = max_cnt

		self.stucking_ratio_best_list = list()
		self.stucking_ratio_better_list = list()
		self.stucking_best_record = list()		
		self.best_reward_list = list()

	def update(self, sample, jid=0):
		self.sample_cnt += 1
		if(not self.sample_buffer):
			sample["vcnt"] = 1
			self.sample_buffer.append(sample)
		else:
			#find all neighbor
			neighbor_list = list()
			neighbor_index_list = list()
			for index, old_sample in enumerate(self.sample_buffer):
				d = self.distance(old_sample, sample, self.distance_type, self.weight)
				if(d <= self.distance_threshold):
					neighbor_list.append(old_sample)
					neighbor_index_list.append(index)
				else:
					pass
				#print(f"distance:{d}")
			#update judgement
			is_update = True
			if(neighbor_list):
				for neighbor in neighbor_list:
					if(neighbor["reward"] >= sample["reward"]): is_update = is_update and False
					else: is_update = is_update and True
				if(is_update):
					pop_list = sorted(neighbor_index_list, reverse = True)
					for index in pop_list:
						self.sample_buffer.pop(index)
					sample["vcnt"] = 1
					self.sample_buffer.append(sample)
				else:
					for neighbor in neighbor_list:
						neighbor["vcnt"] += 1
			else:
				sample["vcnt"] = 1
				self.sample_buffer.append(sample)
			self.is_update = is_update
	
		#### update the stucking_cnt_best
		if(sample["reward"] > self.best_reward):
			self.stucking_cnt_best = 0
			#print(f"$$$$ best sample updating: jid-{jid}|reward-{sample['reward']}|no-{self.sample_cnt}")
			self.stucking_best_record.append([jid,sample['reward'],self.sample_cnt])
		else:
			self.stucking_cnt_best += 1
		if(sample["reward"] > self.better_reward and self.is_update):
			self.stucking_cnt_better = 0
			#print(f"^^^^ better sample updating: jid-{jid}|reward-{sample['reward']}|no-{self.sample_cnt}")
		else:
			self.stucking_cnt_better += 1
		self.stucking_ratio_best = self.stucking_cnt_best/self.max_cnt
		self.stucking_ratio_better = self.stucking_cnt_better/self.max_cnt

		self.best_reward_list.append(self.best_reward)
		self.stucking_ratio_best_list.append(self.stucking_ratio_best)
		self.stucking_ratio_better_list.append(self.stucking_ratio_better)
		if(self.sample_cnt == 7999):
			best_list = numpy.array(self.stucking_ratio_best_list)
			best_csv = pandas.DataFrame(best_list)
			best_csv.to_csv("./record/stucking_ratio_best.csv")

			better_list = numpy.array(self.stucking_ratio_better_list)
			better_csv = pandas.DataFrame(better_list)
			better_csv.to_csv("./record/stucking_ratio_better.csv")

			best_reward_list = numpy.array(self.best_reward_list)
			best_reward_csv = pandas.DataFrame(best_reward_list)
			best_reward_csv.to_csv("./record/best_reward.csv")

			best_record = numpy.array(self.stucking_best_record)
			best_record_csv = pandas.DataFrame(best_record)
			best_record_csv.to_csv("./record/best_record.csv")

		#print(f"status: stucking_cnt_best:{self.stucking_cnt_best}|stucking_cnt_better:{self.stucking_cnt_better}")
		sample_buffer_back = sorted(self.sample_buffer, key = lambda sample:sample["reward"], reverse = True)
		self.best_reward = sample_buffer_back[0]["reward"]
		self.better_reward = sample_buffer_back[min(self.size(), self.sampling_window_lenth)-1]["reward"]

		#### update the distance_threshold
		if(self.adaptive):
			if(self.size() > self.expect_size_ub): 
				self.distance_threshold += self.threshold_step
				self.distance_threshold = min(1, self.distance_threshold)
				back_sample_buffer = copy.deepcopy(self.sample_buffer)
				back_sample_buffer.reverse()
				for sample in back_sample_buffer:
					if(self.is_in_bufffer(sample)): self.fresh(sample)
			elif(self.size() < self.expect_size_lb):
				self.distance_threshold -= self.threshold_step
				self.distance_threshold = max(0, self.distance_threshold)
			else: pass

	def fresh(self, sample):
		if(self.sample_buffer):
			#find all neighbor
			neighbor_list = list()
			neighbor_index_list = list()
			for index, old_sample in enumerate(self.sample_buffer):
				d = self.distance(old_sample, sample, self.distance_type, self.weight)
				#### while updat, the sample self should be excluded (d!=0, old_sample["action_list"] != sample["action_list"])
				#if(d <= self.distance_threshold and d!= 0):
				if(d <= self.distance_threshold and old_sample["action_list"] != sample["action_list"]):
					neighbor_list.append(old_sample)
					neighbor_index_list.append(index)
				else:
					pass
			#update judgement
			is_update = True
			if(neighbor_list):
				for neighbor in neighbor_list:
					#### if samples are with same reward, only reserve one of them, therefore use > rather than >=  
					if(neighbor["reward"] > sample["reward"]):
						is_update = is_update and False
					else: is_update = is_update and True
				if(is_update):
					pop_list = sorted(neighbor_index_list, reverse = True)
					for index in pop_list:
						self.sample_buffer.pop(index)
			
	def acquire(self, atype):
		if(atype == "max"):
			sample_buffer_back = sorted(self.sample_buffer, key = lambda sample:sample["reward"], reverse = True)
			temp_sample = sample_buffer_back[0]
			reward = temp_sample["reward"]
			return temp_sample
		elif(atype == "better"):
			sample_buffer_back = sorted(self.sample_buffer, key = lambda sample:sample["reward"], reverse = True)
			#sample_buffer_back = sorted(self.sample_buffer, key = lambda sample:(sample["reward"]/(sample["vcnt"])**0.5), reverse = True)
			better_sample_cnt = min(int(self.size()), self.sampling_window_lenth)
			
			# print(f"$$$$$$$$$$$$$$$$$$$$$$$$$")
			# for i in range(0, better_sample_cnt):
			# 	print(f"best{i}:{sample_buffer_back[i]['action_list'][4:15]}; r/c:{sample_buffer_back[i]['reward']},{sample_buffer_back[i]['vcnt']}")
			
			if(self.size() > 1):
				sample_index = random.randint(1, better_sample_cnt-1)
			else:
				sample_index = 0
			sample_index = min(sample_index, self.size()-1)
			temp_sample = sample_buffer_back[sample_index]
			return temp_sample			
		elif(atype == "random"):
			use_new_strategy = True

			if(not use_new_strategy):
				#### 20230215 That is the old "potential" strategy
				temp_sample = 0
				temp_sample_metric = 0
				for sample in self.sample_buffer:
					sample_metric = sample["reward"]/(sample["vcnt"]**0.5)
					#sample_metric = 1/(sample["vcnt"]**0.5)
					if(sample_metric > temp_sample_metric): 
						temp_sample_metric = sample_metric
						temp_sample = sample
				#reward = temp_sample["reward"]
				#print(f"random_sample_reward:{reward}")
			else:
				#### 20230215 That is the new "potential" strategy
				for sample in self.sample_buffer:
					sample_metric = sample["reward"]/(sample["vcnt"]**0.5)
					sample["potential"] = sample_metric
				sample_buffer_back = sorted(self.sample_buffer, key = lambda sample:sample["potential"], reverse = True)
				potential_sample_cnt = min(int(self.size()), self.sampling_window_lenth)
				sample_index = random.randint(0, potential_sample_cnt-1)
				sample_index = min(sample_index, self.size()-1)
				temp_sample = sample_buffer_back[sample_index]
				
				'''
				print(f"###########################")
				print(f"buffer size:{self.size()}")
				for cnt in range(0, potential_sample_cnt):
					print(f"random_sample_reward:reward, {sample_buffer_back[cnt]['reward']}; potential, {sample_buffer_back[cnt]['potential']}; cnt, {sample_buffer_back[cnt]['vcnt']}")
				'''
			return temp_sample	

	def distance(self, old_sample, sample, dtype, weight = None):
		distance = 0
		if(dtype == "Euclidean"):
			a = old_sample["obs"]
			b = sample["obs"]
			for i, j in zip(a, b): 
				distance += (i - j)**2
			distance = distance**0.5
			distance = distance / (len(a)**0.5)
		if(dtype == "Hamming"):
			a = old_sample["obs"]
			b = sample["obs"]
			for i, j in zip(a, b):
				if(i != j): distance += 1
			distance = distance / len(a)
		if(dtype == "Weighted_Euclidean"):
			if(weight):
				a = old_sample["obs"]
				b = sample["obs"]
				for i, j, w in zip(a, b, weight): 
					distance += w * (i - j)**2
				distance = distance**0.5
			else:
				print(f"weight list is empty, please input a avaiable weight list")	
				distance = 0
		if(dtype == "Weighted_Hamming"):
			if(weight):
				a = old_sample["obs"]
				b = sample["obs"]
				for i, j, w in zip(a, b, weight): 
					if(i != j): distance += w * 1
			else:
				print(f"weight list is empty, please input a avaiable weight list")	
				distance = 0		

		return float(distance)

	def loadbuffer(self, newbuffer):
		for sample in newbuffer:
			self.update(sample)

	def getbestreward(self):
		reward = 0
		for sample in self.sample_buffer:
			if(sample["reward"] > reward): reward = sample["reward"]
		return reward

	def is_empty(self):
		if(not self.sample_buffer): return True
		else: return False

	def size(self):
		return len(self.sample_buffer)

	def threshold(self):
		return self.distance_threshold

	def print(self):
		dataframe = pandas.DataFrame(self.sample_buffer)
		dataframe.to_csv("./data/data_buffer.csv")

	def getsampleinfo(self):
		self.print()
		obs_list, reward_list, metric_list = list(), list(), list()
		for sample in self.sample_buffer:
			obs_list.append(sample["obs"])
			reward_list.append(sample["reward"])
			metric_list.append(sample["metrics"])
		return obs_list, reward_list, metric_list

	def getstuckingstatus(self):
		return self.stucking_ratio_best, self.stucking_ratio_better

	def getsamplecnt(self):
		return self.sample_cnt

	def is_in_bufffer(self, sample):
		for old_sample in self.sample_buffer:
			if(sample["action_list"] == old_sample["action_list"]): return True
		return False

class simple_warehouse():
	def __init__(self, design_space, metrics_name):
		obs_name_list = list(design_space.get_status().keys())
		obs_lenth = len(obs_name_list)
		metrics_name_list = metrics_name
		metrics_lenth = len(metrics_name_list)
		obs_range = range(0, obs_lenth)
		metrics_range = range(obs_lenth, obs_lenth + metrics_lenth)

		self.sample_buffer = list()
		self.design_space = design_space
		self.obs_name_list = obs_name_list
		self.metrics_name_list = metrics_name_list
		self.obs_range = obs_range
		self.metrics_range = metrics_range

	def append(self, sample):
		self.sample_buffer.append(sample)

	def is_empty(self):
		if(not self.sample_buffer): return True
		else: return False

	def size(self):
		return len(self.sample_buffer)

	def get_metrics_baseline(self):
		data = numpy.array(self.sample_buffer)
		metrics = data[..., self.metrics_range]
		metrics_baseline = metrics.max(axis = 0)
		metrics_average = numpy.mean(metrics, axis = 0)
		metrics_baseline = list(metrics_baseline)
		metrics_average = list(metrics_average)
		return metrics_baseline, metrics_average

	def save_corr_spearman(self, filepath):
		assert(self.sample_buffer)
		data = numpy.array(self.sample_buffer)
		columns = self.obs_name_list + self.metrics_name_list 
		dataframe = pandas.DataFrame(data, columns = columns)
		metrics_corr_table = []
		for metrics_name in self.metrics_name_list:
			metrics_corr_list = []
			for obs_index, obs_name in enumerate(self.obs_name_list):
				metrics = dataframe[metrics_name]
				obs = dataframe[obs_name]
				corr = abs(metrics.corr(obs, method = "spearman"))
				metrics_corr_list.append(corr)
			metrics_corr_table.append(metrics_corr_list)

		corr_table = numpy.array(metrics_corr_table)
		corr_table_dataframe = pandas.DataFrame(corr_table, columns = self.obs_name_list)
		corr_table_dataframe.to_csv(filepath, index = None)

	def load(self, filepath):
		dataframe = pandas.read_csv(filepath)
		data = dataframe.iloc[0::].values
		sample_buffer = data.tolist()
		self.sample_buffer = sample_buffer

	def save(self, filepath):
		data = numpy.array(self.sample_buffer)
		columns = self.obs_name_list + self.metrics_name_list 
		dataframe = pandas.DataFrame(data, columns = columns)
		dataframe.to_csv(filepath, index = None)

	def save_baseline(self, filepath):
		metrics_name_avg_list = list()
		for metrics_name in self.metrics_name_list:
			metrics_name_avg_list.append(metrics_name + "_avg")
		columns = self.obs_name_list + self.metrics_name_list + metrics_name_avg_list
		obs_baseline = self.design_space.get_dimension_upbound()
		metrics_baseline, metrics_average = self.get_metrics_baseline()
		baseline = obs_baseline + metrics_baseline + metrics_average
		data_baseline = numpy.array(baseline).reshape(1, len(baseline))
		dataframe_baseline = pandas.DataFrame(data_baseline, columns = columns)
		dataframe_baseline.to_csv(filepath, index = None)

class warehouse():
	def __init__(self, design_space, config, metrics_name,
		obs_distance_type = "Hamming", metrics_distance_type = "Euclidean", 
		obs_distance_threshold = 0.2, metrics_distance_threshold = 0.2, GEMA = 0.999):
		goal = config.goal
		constraints = config.constraints
		obs_name_list = list(design_space.get_status().keys())
		obs_lenth = len(obs_name_list)
		metrics_name_list = metrics_name
		metrics_lenth = len(metrics_name_list)
		goal_index = obs_lenth + metrics_name_list.index(goal)
		obs_range = range(0, obs_lenth)
		metrics_range = range(obs_lenth, obs_lenth + metrics_lenth)

		self.sample_buffer = list()
		self.design_space = design_space
		self.goal = goal
		self.constraints = constraints
		self.goal_index = goal_index
		self.obs_name_list = obs_name_list
		self.metrics_name_list = metrics_name_list
		self.obs_range = obs_range
		self.metrics_range = metrics_range
		self.obs_distance_type = obs_distance_type
		self.metrics_distance_type = metrics_distance_type
		self.obs_distance_threshold = obs_distance_threshold
		self.metrics_distance_threshold = metrics_distance_threshold
		self.GEMA = GEMA
		self.baseline = list()

	def append(self, sample):
		self.sample_buffer.append(sample)

	def update(self, sample):
		assert(self.baseline)
		baseline = self.baseline
		if(not self.sample_buffer):
			sample.append(1)
			self.sample_buffer.append(sample)
		else:
			#find all neighbor
			obs_neighbor_list = list()
			obs_neighbor_index_list = list()
			for index, old_sample in enumerate(self.sample_buffer):
				obs_d = self.distance(old_sample, sample, baseline, "obs", self.obs_distance_type)
				if(obs_d <= self.obs_distance_threshold):
					obs_neighbor_list.append(old_sample)
					obs_neighbor_index_list.append(index)
				else:
					pass
			metrics_neighbor_list = list()
			metrics_neighbor_index_list = list()
			for index, old_sample in enumerate(self.sample_buffer):
				metrics_d = self.distance(old_sample, sample, baseline, "metrics", self.metrics_distance_type)
				if(metrics_d <= self.metrics_distance_threshold):
					metrics_neighbor_list.append(old_sample)
					metrics_neighbor_index_list.append(index)
				else:
					pass
			#update judgement
			obs_is_update = True
			metrics_is_update = True
			if(obs_neighbor_list or metrics_neighbor_list):
				if(obs_neighbor_list):
					for obs_neighbor in obs_neighbor_list:
						if(obs_neighbor[self.goal_index] >= sample[self.goal_index]): obs_is_update = obs_is_update and False
						else: obs_is_update = obs_is_update and True
				if(metrics_neighbor_list):
					for metrics_neighbor in metrics_neighbor_list:
						if(metrics_neighbor[self.goal_index] >= sample[self.goal_index]): metrics_is_update = obs_is_update and False
						else: metrics_is_update = metrics_is_update and True				
				if(obs_is_update and metrics_is_update):
					neighbor_index_list = list(set(obs_neighbor_index_list).union(metrics_neighbor_index_list))
					pop_list = sorted(neighbor_index_list, reverse = True)
					for index in pop_list:
						self.sample_buffer.pop(index)
					sample.append(1)
					self.sample_buffer.append(sample)
				else:
					if(obs_neighbor_list):
						for neighbor in obs_neighbor_list:
							neighbor[-1] += 1
					if(metrics_neighbor_list):
						for neighbor in metrics_neighbor_list:
							neighbor[-1] += 1						
			else:
				sample.append(1)
				self.sample_buffer.append(sample)

	def distance(self, old_sample, sample, baseline, itype, dtype, weight = None):
		distance = 0
		old_sample = numpy.array(old_sample)
		sample = numpy.array(sample)
		baseline = numpy.array(baseline)
		if(dtype == "Euclidean"):
			if(itype == "obs"):
				a = old_sample[self.obs_range]/baseline[self.obs_range]
				b = sample[self.obs_range]/baseline[self.obs_range]
			elif(itype == "metrics"):
				a = old_sample[self.metrics_range]/baseline[self.metrics_range]
				b = sample[self.metrics_range]/baseline[self.metrics_range]	
			for i, j in zip(a, b): 
				distance += (i - j)**2
			distance = distance**0.5
			distance = distance / (len(a)**0.5)
		if(dtype == "Hamming"):
			if(itype == "obs"):
				a = old_sample[self.obs_range]/baseline[self.obs_range]
				b = sample[self.obs_range]/baseline[self.obs_range]
			elif(itype == "metrics"):
				a = old_sample[self.metrics_range]/baseline[self.metrics_range]
				b = sample[self.metrics_range]/baseline[self.metrics_range]
			for i, j in zip(a, b):
				if(i != j): distance += 1
			distance = distance / len(a)
		if(dtype == "Weighted_Euclidean"):
			if(weight):
				if(itype == "obs"):
					a = old_sample[self.obs_range]/baseline[self.obs_range]
					b = sample[self.obs_range]/baseline[self.obs_range]
				elif(itype == "metrics"):
					a = old_sample[self.metrics_range]/baseline[self.metrics_range]
					b = sample[self.metrics_range]/baseline[self.metrics_range]
				for i, j, w in zip(a, b, weight): 
					distance += w * (i - j)**2
				distance = distance**0.5
				distance = distance / (len(a)**0.5)
			else:
				print(f"weight list is empty, please input a avaiable weight list")	
				distance = 0		

		return float(distance)

	def is_empty(self):
		if(not self.sample_buffer): return True
		else: return False

	def size(self):
		return len(self.sample_buffer)

	def load_baseline(self, filepath):
		dataframe_baseline = pandas.read_csv(filepath)
		data_baseline = dataframe_baseline.values
		baseline = data_baseline.tolist()[0]
		self.baseline = baseline

	def load(self, filepath):
		dataframe = pandas.read_csv(filepath)
		data = dataframe.iloc[0::].values
		sample_buffer = data.tolist()
		self.sample_buffer = sample_buffer

	def save(self, filepath):
		data = numpy.array(self.sample_buffer)
		columns = self.obs_name_list + self.metrics_name_list + ["vcnt"]
		dataframe = pandas.DataFrame(data, columns = columns)
		dataframe.to_csv(filepath, index = None)

	def create_buffer(self):
		assert(self.baseline)
		newbuffer = list()
		baseline = self.baseline
		obs_baseline = list(numpy.array(baseline)[self.obs_range])
		goal_baseline = baseline[self.goal_index]
		for sample in self.sample_buffer:
			new_sample = dict()

			obs_list = list(numpy.array(sample)[self.obs_range])
			metrics_list = list(numpy.array(sample)[self.metrics_range])

			new_sample["obs"] = list(numpy.array(obs_list)/numpy.array(obs_baseline))

			metrics = dict()
			for metric, metrics_name in zip(metrics_list, self.metrics_name_list):
				metrics[metrics_name] = metric
			new_sample["metrics"] = metrics

			constraints_name_list = self.constraints.get_name_list()
			for constraints_name in constraints_name_list:
				self.constraints.multi_update(metrics)
			objectvalue = metrics[self.goal] / goal_baseline
			reward = 1 / (objectvalue * self.constraints.get_punishment())
			T = len(self.obs_range)
			reward_list = [0 for step in range(T)]
			reward_list[-1] = reward
			return_g = 0
			return_list = list()
			for t in range(T):
				return_g = reward_list[T-1-t] + self.GEMA * return_g
				return_list.append(return_g)
			return_list.reverse()			
			new_sample["reward"] = reward
			new_sample["return_list"] = return_list	

			action_list = list()
			for obs, dimension in zip(obs_list, self.design_space.dimension_box):
				action = dimension.sample_box.index(round(obs,1))
				action_list.append(action)
			new_sample["action_list"] = action_list

			newbuffer.append(new_sample)
		return newbuffer

