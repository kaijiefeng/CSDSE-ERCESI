import torch
import numpy
import random
import pdb
import os
import time
import pandas

import numpy as np
from sklearn import manifold
import matplotlib.pyplot as plt

#from evaluation import evaluation_function
from evaluation_maestro import evaluation_maestro

def find_divisor(number):
	number = int(number)
	half =number + 1
	divisor_list = []
	for divisor in range(1, half):
		if(number % divisor == 0): divisor_list.append(divisor)
		else: pass
	return divisor_list

class dimension_discrete():
	def __init__(self, name, default_value, step, rrange, frozen = False, model = {"name":"normal", "param":0.4}):
		'''
		"name"-string
		"default_value"-int
		"step"-int
		"rrange"-[low-int,high-int]
		'''
		self.name = name
		self.default_value = default_value
		self.current_value = default_value
		self.step = step
		self.frozen = frozen
		self.model = model

		assert(rrange[0] <= rrange[-1])
		self.rrange = [rrange[0], rrange[-1]]
		self.sample_box = []

		if(self.step > 0):
			self.scale = (self.rrange[-1] - self.rrange[0])//self.step + 1

			self.default_index = int((default_value - self.rrange[0])//self.step)
			self.current_index = self.default_index
			'''
			sample_box offers the sample space for discrete dimension
			every item in sample_box is a avaiable value for that dimension
			NOW: thr range of sample_box is [rrange[0] ~ rrange[0]+step*(scale-1) rather than [rrange[0] ~ rrange[-1]]
			that's not a big deal
			'''
			
			for idx in range(int(self.scale)):
				self.sample_box.append(round(self.rrange[0]+idx*step,1))
		else:
			self.scale = len(rrange)
			self.default_index = rrange.index(default_value)
			self.current_index = self.default_index
			self.sample_box = rrange

	def set(self, sample_index):
		assert (sample_index >= 0 and sample_index <=self.scale-1)
		self.current_index = sample_index
		self.current_value = self.sample_box[sample_index]
	def set_max(self):
		self.current_index = int(self.scale-1)
		self.current_value = self.sample_box[self.current_index]
	def original_set(self, sample_index):
		assert (sample_index >= 0 and sample_index <=self.scale-1)
		self.default_index = sample_index
		self.default_value = self.sample_box[sample_index]
	def reset(self):
		self.current_index = self.default_index
		self.current_value = self.default_value
	def get_name(self):
		return self.name
	def get_scale(self):
		return self.scale
	def get_range_upbound(self):
		return self.rrange[-1]
	def sample(self, sample_index):
		assert (sample_index >= 0 and sample_index <=self.scale-1)
		if(self.frozen == False):
			self.current_index = sample_index
			self.current_value = self.sample_box[sample_index]
		return self.current_value
	def get_current_index(self):
		return self.current_index
	def get_current_value(self):
		return self.current_value
	def get_sample_box(self):
		return self.sample_box
	def froze(self):
		self.frozen = True
	def release(self):
		self.frozen = False
	def get_model(self):
		return self.model

	#### new function 2021/07/26
	def get_norm_current_value(self):
		return self.get_current_value()/self.get_range_upbound()

class design_space():
	def __init__(self):
		'''
		dimension_box is a list of dict which is consist of two item, "name":str and "dimension":dimension_discrete
		'''
		self.dimension_box = []
		self.lenth = 0
		self.scale = 1

		self.const_lenth = 0
		self.dynamic_lenth = 0

		self.layer_name = []

		self.HRP, self.LRP = [],[]

		self.upbound = {}
	def append(self,dimension_discrete):
		self.dimension_box.append(dimension_discrete)
		self.lenth = self.lenth + 1
		self.scale = self.scale * dimension_discrete.get_scale()
		self.upbound[dimension_discrete.get_name()] = dimension_discrete.get_range_upbound()
	def get_status(self):
		'''
		status is a dict class that can be used for matching of dimension "name":"dimension_value"
		'''
		status = dict()
		for item in self.dimension_box:
			status[item.get_name()] = item.get_current_value()
		return status
	def get_current_status(self, dimension_index):
		status = dict()
		item = self.dimension_box[dimension_index]
		status[item.get_name()] = item.get_current_value()
		return status
	def get_compact_status(self, dimension_index):
		status = dict()
		if(dimension_index < self.const_lenth):
			layer_index = 0
			layer = "Hardware"
			temp_layer = self.layer_name[layer_index]
		else:
			layer_index = int((dimension_index - self.const_lenth)/self.dynamic_lenth)
			layer = self.layer_name[layer_index]

		const_range = range(0, self.const_lenth)
		dynamic_range = range(self.const_lenth+layer_index*self.dynamic_lenth, self.const_lenth+(layer_index+1)*self.dynamic_lenth)
		if(layer == "Hardware"):
			for dindex in const_range:
				item = self.dimension_box[dindex]
				status[item.get_name()] = item.get_current_value()
			for dindex in dynamic_range:
				item = self.dimension_box[dindex]
				status[item.get_name()] = 0
		else:
			for dindex in const_range:
				item = self.dimension_box[dindex]
				status[item.get_name()] = item.get_current_value()
			for dindex in dynamic_range:	
				item = self.dimension_box[dindex]
				status[item.get_name()] = item.get_current_value()
		return status

	def get_status_value(self):
		status_value = list()
		for item in self.dimension_box:
			status_value.append(item.get_current_value())
		return status_value
	def get_action_list(self):
		action_list = list()
		for item in self.dimension_box:
			action_list.append(item.get_current_index())
		return action_list
	def print_status(self):
		for item in self.dimension_box:
			print(item.get_name(),item.get_current_value())
	def sample_one_dimension(self, dimension_index, sample_index):
		assert (dimension_index >= 0 and dimension_index <= self.lenth-1)
		self.dimension_box[dimension_index].sample(sample_index)
		#return self.get_status()
	def set_one_dimension(self, dimension_index, sample_index):
		assert (dimension_index >= 0 and dimension_index <= self.lenth-1)
		self.dimension_box[dimension_index].set(sample_index)
		#return self.get_status()		
	def status_set(self, best_action_list):
		for dimension, action in zip(self.dimension_box, best_action_list):
			dimension.set(action)
		return self.get_status()
	def original_status_set(self, best_action_list):
		for dimension, action in zip(self.dimension_box, best_action_list):
			dimension.original_set(action)
		return self.get_status()
	def status_reset(self):
		for dimension in self.dimension_box:
			dimension.reset()
		return self.get_status()
	def get_lenth(self):
		return self.lenth
	def get_scale(self):
		return self.scale
	def get_dimension_current_index(self, dimension_index):
		return self.dimension_box[dimension_index].get_current_index()
	def get_dimension_scale(self, dimension_index):
		return self.dimension_box[dimension_index].get_scale()
	def get_dimension_sample_box(self, dimension_index):
		return self.dimension_box[dimension_index].sample_box
	def froze_one_dimension(self, dimension_index):
		self.dimension_box[dimension_index].froze()
	def release_one_dimension(self, dimension_index):
		self.dimension_box[dimension_index].release()
	def froze_dimension(self, dimension_index_list):
		for index in dimension_index_list:
			self.froze_one_dimension(index)
	def release_dimension(self, dimension_index_list):
		for index in dimension_index_list:
			self.release_one_dimension(index)
	def get_dimension_model(self, dimension_index):
		return self.dimension_box[dimension_index].get_model()
	def get_dimension_upbound(self):
		dimension_upbound = list()
		for item in self.dimension_box:
			dimension_upbound.append(item.get_range_upbound())
		return dimension_upbound
	def get_dimension_scale_list(self, has_upbound = True):
		dimension_scale_list = list();
		for item in self.dimension_box:
			if(has_upbound): dimension_scale_list.append(item.get_scale())
			else: dimension_scale_list.append(item.get_scale() - 1)
		return dimension_scale_list

	#### new function, for ppo, require numpy and torch
	def get_obs(self):
		obs_list = list()
		for item in self.dimension_box:
			obs_list.append(item.get_norm_current_value())
		obs = numpy.array(obs_list)
		#obs = torch.from_numpy(obs)
		return obs

	def get_compact_obs(self, dimension_index):
		obs_list = list()
		if(dimension_index < self.const_lenth):
			layer_index = 0
			layer = "Hardware"
			temp_layer = self.layer_name[layer_index]
		else:
			layer_index = int((dimension_index - self.const_lenth)/self.dynamic_lenth)
			layer = self.layer_name[layer_index]

		const_range = range(0, self.const_lenth)
		dynamic_range = range(self.const_lenth+layer_index*self.dynamic_lenth, self.const_lenth+(layer_index+1)*self.dynamic_lenth)
		if(layer == "Hardware"):
			for dindex in const_range:
				item = self.dimension_box[dindex]
				obs_list.append(item.get_norm_current_value())
			for dindex in dynamic_range:
				item = self.dimension_box[dindex]
				obs_list.append(0)
		else:
			for dindex in const_range:
				item = self.dimension_box[dindex]
				obs_list.append(item.get_norm_current_value())
			for dindex in dynamic_range:	
				item = self.dimension_box[dindex]
				obs_list.append(item.get_norm_current_value())
		obs = numpy.array(obs_list)
		return obs

	def get_obs_dlrm(self):
		obs_list = list()
		for item in self.dimension_box:
			if item.get_model()["name"] == "normal":
				obs_list.append(item.get_norm_current_value())
			elif item.get_model()["name"] == "one_hot":
				obs_list.append(item.get_current_index())
		obs = numpy.array(obs_list)
		#obs = torch.from_numpy(obs)
		return obs
	
	def get_obs_den(self):
		obs_list = list()
		for item in self.dimension_box:
			if item.get_model()["name"] == "normal":
				obs_list.append(item.get_current_index())
		obs = numpy.array(obs_list)
		#obs = torch.from_numpy(obs)
		return obs

	def get_obs_spa(self):
		obs_list = list()
		for item in self.dimension_box:
			if item.get_model()["name"] == "one_hot":
				obs_list.append(item.get_current_index())
		obs = numpy.array(obs_list)
		#obs = torch.from_numpy(obs)
		return obs

	#### for TLDSE
	def get_den_spa(self):
		den_idx = list()
		spa_idx = list()
		spa_shape = list()
		for index, dimension in enumerate(self.dimension_box):
			if dimension.get_model()["name"] == "normal": 
				den_idx.append(index)
			elif dimension.get_model()["name"] == "one_hot": 
				spa_idx.append(index)
				spa_shape.append(dimension.get_scale())
			else: pass
		return den_idx, spa_idx, spa_shape

	#### for heuristic
	def set_max_all(self):
		for item in self.dimension_box:
			item.set_max()

	#### for CSDSE
	def corr_analysis(self, filepath, threshold = 0.1):
		corr_table_dataframe = pandas.read_csv(filepath)
		corr_table = corr_table_dataframe.iloc[0::].values.tolist()
		corr_index_table = []
		for corr_list in corr_table:
			corr_index_list = []
			for index, corr in enumerate(corr_list):
				if(corr >= threshold): corr_index_list.append(index)
			corr_index_table.append(corr_index_list)

		PMASK = list(range(0, self.lenth))
		HRP = list()
		for corr_index_list in corr_index_table:
			HRP = list(set(HRP).union(corr_index_list))
		LRP = list(set(PMASK) ^ set(HRP))
		HRP.sort()
		LRP.sort()
		self.HRP, self.LRP = HRP, LRP

		HRP_weight = corr_table_dataframe.iloc[:,HRP].mean(axis=0)
		HRP_weight_sum = HRP_weight.sum(axis=0)
		HRP_weight = HRP_weight / HRP_weight_sum
		HRP_weight = HRP_weight.tolist()
		HRP_weight = zip(HRP, HRP_weight)

		HRP_ratio = len(HRP) / self.lenth
		print(f"HRP weight:{HRP, HRP_weight}")
		print(f"HRP_ratio:{HRP_ratio}")

		return HRP, LRP, HRP_weight

	def compress(self, nnmodel, period):
		#### HRP&LRP
		if(not self.HRP or not self.LRP):
			filepath = "./data/corr_table_{}.csv".format(nnmodel)
			assert(os.path.exists(filepath))
			self.corr_analysis(filepath)

		interval = 2
		if(period >= 0):
			if(period % interval != 0):
				self.release_dimension(self.HRP)
				self.froze_dimension(self.LRP)
			else:
				self.froze_dimension(self.HRP)
				self.release_dimension(self.LRP)
		else:
			pass

def create_space_maestro(model, is_adaptive = True, is_const = False, target = "largeedge"):
	## get the model from model file
	if(model == 'VGG16'): model_filename = './desc/model/vgg16_model.m'
	elif(model == 'MobileNetV2'): model_filename = './desc/model/MobileNetV2_model.m'
	elif(model == 'MnasNet'): model_filename = './desc/model/mnasnet_model.m'
	elif(model == 'ResNet50'): model_filename = './desc/model/Resnet50_model.m'
	#elif(model == 'Transformer'): model_filename = './desc/model/Transformer_Complete_model.m'
	elif(model == 'Transformer'): model_filename = './desc/model/Transformer_Complete_model_littleRS.m'
	elif(model == 'GNMT'): model_filename = './desc/model/gnmt_model.m'
	else: pass
	## get the layer name and type from model file
	layer_list = []
	type_list = []
	C_list, K_list, X_list, Y_list = [], [], [], []
	dimension_list = []
	stride_list = []
	with open(model_filename, 'r') as mdfile:
		lines = mdfile.readlines()
		for line in lines:
			if(line.find('Layer') != -1):
				start = line.find('Layer') + len('Layer ')
				end = line.find(' {')
				layer_list.append(line[start:end])
			if(line.find('Type') != -1):
				start = line.find('Type') + len('Type: ')
				if(line.find('//') != -1):
					end = start + line[start:].find(' ')
				else:
					end = line.find('\n')
				type_list.append(line[start:end])
			if(line.find('Stride') != -1):
				start = line.find('X') + len('X: ')
				end = start + line[start:].find(',')
				S_X = int(line[start:end])
				start = line.find('Y') + len('Y: ')
				end = start + line[start:].find(' }')
				S_Y = int(line[start:end])
				stride_list.append(S_X)
			if(line.find('Dimensions') != -1):
				start = line.find('R') + len('R: ')
				end = start + line[start:].find(',')
				R = int(line[start:end])
				start = line.find('S') + len('S: ')
				end = start + line[start:].find(',')
				S = int(line[start:end])
				start = line.find('C') + len('C: ')
				end = start + line[start:].find(',')
				C = int(line[start:end])
				C_list.append(int(line[start:end]))
				start = line.find('K') + len('K: ')
				end = start + line[start:].find(',')
				K = int(line[start:end])
				K_list.append(int(line[start:end]))
				
				if(line.find('X: ') != -1):
					start = line.find('X') + len('X: ')
				elif(line.find('X:') != -1):
					start = line.find('X') + len('X:')
				#start = line.find('X') + len('X: ')
				end = start + line[start:].find(' }')
				X = int(line[start:end])
				X_ = int((X - S) / S_X) + 1 
				if(line.find('Y: ') != -1):
					start = line.find('Y') + len('Y: ')
				elif(line.find('Y:') != -1):
					start = line.find('Y') + len('Y:')
				#start = line.find('Y') + len('Y: ')
				end = start + line[start:].find(',')
				Y = int(line[start:end])
				Y_ = int((Y - R) / S_Y) + 1 
				Y_list.append(Y_)

				if(Y_ < 0): print(f"layer:{layer_list[-1]}")

				dimension = []
				dimension.append(C)
				dimension.append(K)
				dimension.append(X_)
				dimension.append(Y_)
				dimension.append(R)
				dimension.append(S)
				dimension_list.append(dimension)

	## layers with the same shapes will be grouped into blocks for uniformly optimization
	dimension_set = list()
	block_list = list()
	is_clustered = True
	if(is_clustered):
		for dimension in dimension_list:
			if(dimension not in dimension_set): dimension_set.append(dimension)	
		for dimension_index, dimension in enumerate(dimension_list):
			block_index = dimension_set.index(dimension)
			block_list.append(block_index)
		print(f"layer leth (after being clustered): {len(dimension_set)}")
	else:
		dimension_set = dimension_list
		for dimension_index, dimension in enumerate(dimension_list):
			block_index = dimension_index
			block_list.append(block_index)
		print(f"layer leth (without being clustered): {len(dimension_set)}")

	## initialize the design space
	DSE_action_space = design_space()

	## define parameters
	l1_size = dimension_discrete(
		name = 'l1_size',
		default_value = 16000,
		step = 0,
		rrange = [128, 256, 512, 1024, 2048, 4096, 8192, 16000, 32000, 64000, 128000, 256000, 512000, 1024000, 2048000]
	)
	DSE_action_space.append(l1_size)
	l2_size = dimension_discrete(
		name = 'l2_size',
		default_value = 512000,
		step = 0,
		rrange = [16000, 32000, 64000, 128000, 256000, 512000, 1024000, 2048000, 4096000, 8192000, 16000000, 32000000]
	)

	#### on NVDLA, max onchip bandwidth is 128B*2 = 256 * 8b 
	DSE_action_space.append(l2_size)
	if(target == "largeedge" or target == "cloud"):
		noc_bw = dimension_discrete(
			name = 'noc_bw',
			default_value = 256,
			step = 0,
			rrange = [256]
		)
		DSE_action_space.append(noc_bw)
		#### on NVDLA, offchip bandwidth is 512b = 64*8b
		offchip_bw = dimension_discrete(
			name = 'offchip_bw',
			default_value = 64,
			step = 0,
			rrange = [64]
		)
		DSE_action_space.append(offchip_bw)
	elif(target == "smalledge"):
		#### on eyeriss, max onchip bandwidth is 144b = 18*8b 
		noc_bw = dimension_discrete(
			name = 'noc_bw',
			default_value = 18,
			step = 0,
			rrange = [18]
		)
		DSE_action_space.append(noc_bw)
		#### on eyeriss, offchip bandwidth is 64b = 8*8b
		offchip_bw = dimension_discrete(
			name = 'offchip_bw',
			default_value = 8,
			step = 0,
			rrange = [8]
		)
		DSE_action_space.append(offchip_bw)

	if(not is_const):
		is_timecost_test = False
		if(is_timecost_test):
			dim_num = dimension_discrete(
				name = 'dim_num',
				default_value = 3,
				step = 0,
				rrange = [3],
				model = {"name":"one_hot", "param":0.1}
			)
		else:
			dim_num = dimension_discrete(
				name = 'dim_num',
				default_value = 3,
				step = 1,
				rrange = [1,3],
				model = {"name":"one_hot", "param":0.1}
			)
		DSE_action_space.append(dim_num)
		dim_out = dimension_discrete(
			name = 'dim_out',
			default_value = 16,
			step = 2,
			rrange = [2, 48]
			#step = 0,
			#rrange = [2,4,8,16,32,64,128,256,512]
		)
		DSE_action_space.append(dim_out)
		dim_mid = dimension_discrete(
			name = 'dim_mid',
			default_value = 16,
			step = 2,
			rrange = [2,48]
			#step = 0,
			#rrange = [2,4,8,16,32,64,128,256,512]
		)
		DSE_action_space.append(dim_mid)
		dim_in = dimension_discrete(
			name = 'dim_in',
			default_value = 16,
			step = 2,
			rrange = [2,48]
			#step = 0,
			#rrange = [2,4,8,16,32,64,128,256,512]
		)
		DSE_action_space.append(dim_in)
		p_name_list = ['c', 'k', 'x', 'y', 'r', 's']
		for p_name in p_name_list:
			p = dimension_discrete(
				name = 'p_{}'.format(p_name),
				default_value = 1,
				step = 1, 
				rrange = [1,6],
				model = {"name":"one_hot", "param":0.1}
			)
			DSE_action_space.append(p)
		if(is_adaptive):
			for dimension_index, dimension in enumerate(dimension_set):
				layer = dimension_index
				C, K, X, Y = dimension[0], dimension[1], dimension[2], dimension[3]
				o_name_list = ['c_out', 'k_out', 'x_out', 'y_out', 'r_out', 's_out',\
				'c_mid', 'k_mid', 'x_mid', 'y_mid', 'r_mid', 's_mid',\
				'c_in', 'k_in', 'x_in', 'y_in', 'r_in', 's_in']
				tc_name_list = ['c_d1','c_d2','c_d3']
				tk_name_list = ['k_d1','k_d2','k_d3']
				tx_name_list = ['x_d1','x_d2','x_d3']
				ty_name_list = ['y_d1','y_d2','y_d3']

				for o_name in o_name_list:
					o = dimension_discrete(
						name = 'o_{}_{}'.format(o_name, layer),
						default_value = 1,
						step = 1, 
						rrange = [1,6],
						model = {"name":"one_hot", "param":0.1}
					)
					DSE_action_space.append(o)
				for tc_name in tc_name_list:
					if(tc_name != 'c_d3'):
						tc = dimension_discrete(
							name = 't_{}_{}'.format(tc_name, layer),
							default_value = 1,
							step = 0, 
							rrange = find_divisor(C)
						)
					else:
						tc = dimension_discrete(
							name = 't_{}_{}'.format(tc_name, layer),
							default_value = 1,
							step = 0, 
							rrange = [1]
						)						
					DSE_action_space.append(tc)
				for tk_name in tk_name_list:
					if(tk_name != 'k_d3'):
						tk = dimension_discrete(
							name = 't_{}_{}'.format(tk_name, layer),
							default_value = 1,
							step = 0, 
							rrange = find_divisor(K)
						)
					else:
						tk = dimension_discrete(
							name = 't_{}_{}'.format(tk_name, layer),
							default_value = 1,
							step = 0, 
							rrange = [1]
						)
					DSE_action_space.append(tk)
				for tx_name in tx_name_list:
					if(tx_name != 'x_d3'):
						tx = dimension_discrete(
							name = 't_{}_{}'.format(tx_name, layer),
							default_value = 1,
							step = 0, 
							rrange = find_divisor(X)
						)
					else:
						tx = dimension_discrete(
							name = 't_{}_{}'.format(tx_name, layer),
							default_value = 1,
							step = 0, 
							rrange = [1]
						)
					DSE_action_space.append(tx)
				for ty_name in ty_name_list:
					if(ty_name != 'y_d3'):
						ty = dimension_discrete(
							name = 't_{}_{}'.format(ty_name, layer),
							default_value = 1,
							step = 0, 
							rrange = find_divisor(Y)
						)
					else:
						ty = dimension_discrete(
							name = 't_{}_{}'.format(ty_name, layer),
							default_value = 1,
							step = 0, 
							rrange = [1]
						)							
					DSE_action_space.append(ty)	
	print(f"lenth:{DSE_action_space.get_lenth()}")
	DSE_action_space.const_lenth = 8 + 6 # 8 for hardware, 6 for parallelism
	DSE_action_space.dynamic_lenth = 3*6 + 3*4 #3*6 for array order, 3*4 for array tiling
	DSE_action_space.layer_name = layer_list
	DSE_action_space.type_list = type_list
	DSE_action_space.dimension_list = dimension_list
	DSE_action_space.stride_list = stride_list
	DSE_action_space.block_list = block_list

	return DSE_action_space			

class environment_maestro():
	def __init__(self, algo, iindex, config, test = False, delay_reward = True):
		self.nnmodel = config.nnmodel
		self.target = config.target
		self.goal = config.goal
		self.constraints = config.constraints
		self.baseline = config.baseline
		self.iindex = iindex
		self.algo = algo
		self.test = test
		self.delay_reward = delay_reward
		self.pid = os.getpid()
				
		self.best_objectvalue = 1000
		self.best_objectvalue_list = list()
		self.multiobjecvalue_list = list()

		self.design_space = create_space_maestro(self.nnmodel, target = self.target)
		self.evaluation = evaluation_maestro(self.iindex, self.nnmodel, self.pid, self.design_space)

		self.design_space_dimension = self.design_space.get_lenth()
		self.action_dimension_list = self.design_space.get_dimension_scale_list(has_upbound = True)
		self.action_limit_list = self.design_space.get_dimension_scale_list(has_upbound = False)

		self.const_lenth = self.design_space.const_lenth
		self.dynamic_lenth = self.design_space.dynamic_lenth
		self.compact_obs_action_dimension_list = list()
		for index in range(self.const_lenth + self.dynamic_lenth):
			self.compact_obs_action_dimension_list.append(self.action_dimension_list[index])
		
	def reset(self):
		self.design_space.status_reset()
		#return self.design_space.get_obs()
		return self.design_space.get_compact_obs(0)

	def step(self, step, act, deterministic=False):
		if(deterministic):
			if(torch.is_tensor(act)): act = torch.argmax(act, dim=-1).item()
			else: act = torch.argmax(torch.as_tensor(act).view(-1), dim=-1).item()
		else:
			if(self.algo == "SAC"):
				if(torch.is_tensor(act)): 
					act = torch.softmax(act, dim = -1)
					act = int(act.multinomial(num_samples = 1).data.item())
				if(isinstance(act, numpy.ndarray)):
					act = torch.as_tensor(act, dtype=torch.float32).view(-1)
					act = torch.softmax(act, dim = -1)
					act = int(act.multinomial(num_samples = 1).data.item())
			elif(self.algo == "PPO"):
				pass
			elif(self.algo == "happo" or self.algo == "hasac"):
				pass
		
		self.design_space.sample_one_dimension(step, act)
		#obs = self.design_space.get_obs()
		obs = self.design_space.get_compact_obs(step)

		if(step < (self.design_space.get_lenth() - 1)):
			not_done = 1
		else:
			not_done = 0

		if(not_done):
			if(self.delay_reward):
				reward = float(0)
			else:
				all_status = self.design_space.get_status()
				metrics = self.evaluation.evaluate(all_status)
				if(metrics != None):
					self.constraints.multi_update(metrics)
					objectvalue = metrics[self.goal] / self.baseline[self.goal]
					reward = 1 / (objectvalue * self.constraints.get_punishment())
				else:
					reward = 0

				print(f"objectvalue:{objectvalue}, reward:{reward}", end = '\r')

				if(not self.test):
					if(objectvalue < self.best_objectvalue and self.constraints.is_all_meet()):
						self.best_objectvalue = objectvalue
					self.best_objectvalue_list.append(self.best_objectvalue)
					self.multiobjecvalue_list.append([metrics["latency"], metrics["energy"]])
		else:
			all_status = self.design_space.get_status()
			metrics = self.evaluation.evaluate(all_status)
			if(metrics != None):
				self.constraints.multi_update(metrics)
				objectvalue = metrics[self.goal] / self.baseline[self.goal]
				reward = 1 / (objectvalue * self.constraints.get_punishment())
			else:
				reward = 0

			print(f"objectvalue:{objectvalue}, reward:{reward}", end = '\r')

			if(not self.test):
				if(objectvalue < self.best_objectvalue and self.constraints.is_all_meet()):
					self.best_objectvalue = objectvalue
				self.best_objectvalue_list.append(self.best_objectvalue)
				self.multiobjecvalue_list.append([metrics["latency"], metrics["energy"]])

		done = not not_done

		return obs, reward, done, {}

	def sample(self, step):
		idx = random.randint(0, self.design_space.get_dimension_scale(step)-1)
		pi = torch.zeros(int(self.design_space.get_dimension_scale(step)))
		pi[idx] = 1
		return pi

def tsne3D(vector_list, reward_list, method):
	action_array = np.array(vector_list)
	reward_continue_array = np.array(reward_list)

	tsne = manifold.TSNE(n_components = 2, init = "pca", random_state = 501)
	print(f"Start to load t-SNE")
	x_tsne = tsne.fit_transform(action_array)

	x_min, x_max = x_tsne.min(0), x_tsne.max(0)
	x_norm = (x_tsne - x_min)/(x_max - x_min)
	#pdb.set_trace()

	fig_3D = plt.figure()
	tSNE_3D = plt.axes(projection = '3d')
	tSNE_3D.scatter3D(x_norm[:, 0], x_norm[:, 1], reward_continue_array, c = reward_continue_array, vmax = 20, cmap = "rainbow", alpha = 0.5)
	#tSNE_3D.scatter3D(x_norm[:, 0], x_norm[:, 1], reward_continue_array, c = reward_continue_array, cmap = "rainbow", alpha = 0.5)
	tSNE_3D.set_xlabel("x")
	tSNE_3D.set_ylabel("y")
	tSNE_3D.set_zlabel("Reward")
	tSNE_3D.set_zlim((0, 20))
	tSNE_3D.set_zticks([0,5,10,15,20])	
	fname = method + "_" + "tSEN_3D" + ".png"
	fig_3D.savefig(fname, format = "png")

def tsne2D(vector_list, reward_list, method, has_interval=False, interval=1000):
	import matplotlib
	print(f"font:{matplotlib.matplotlib_fname()}")
	action_array = np.array(vector_list)
	reward_continue_array = np.array(reward_list)

	tsne = manifold.TSNE(n_components = 2, init = "pca", random_state = 1)
	print(f"Start to load t-SNE")
	x_tsne = tsne.fit_transform(action_array)

	x_min, x_max = x_tsne.min(0), x_tsne.max(0)
	x_norm = (x_tsne - x_min)/(x_max - x_min)
	r_max = reward_continue_array.max(0)
	#pdb.set_trace()

	print(f"Painting...")
	x = x_norm
	r = reward_continue_array
	if(not has_interval):
		fig_2D = plt.figure(dpi=600)
		tSNE_2D = plt.axes()
		tSNE_2D.scatter(x[:, 0], x[:, 1], c = r, vmax = r_max, cmap = "rainbow", s=5, alpha = 0.1)
		#tSNE_3D.scatter3D(x_norm[:, 0], x_norm[:, 1], reward_continue_array, c = reward_continue_array, cmap = "rainbow", alpha = 0.5)
		tSNE_2D.set_xlabel("x")
		tSNE_2D.set_ylabel("y")
		fname = "./record/tSNE2D/{}_tSNE2D.png".format(method)
		fig_2D.savefig(fname, format = "png")
	else:
		#marklist = [".", "^", "s", "p", "*", "+", "x", "D"]
		n = int(len(vector_list)/interval)
		for i in range(0, n):
			fig_2D = plt.figure(dpi=600)
			tSNE_2D = plt.axes()
			#n_mark = i%len(marklist)
			istart = i*interval
			iend = (i+1)*interval
			tSNE_2D.scatter(x[istart:iend, 0], x[istart:iend, 1], c = r[istart:iend], vmax = r_max, cmap = "rainbow", s=20, alpha = 0.3)
			tSNE_2D.set_xlabel("x", fontdict={"family":"Times New Roman", "size":15})
			tSNE_2D.set_xlim(0,1)
			tSNE_2D.set_ylabel("y", fontdict={"family":"Times New Roman", "size":15})
			tSNE_2D.set_ylim(0,1)
			tSNE_2D.tick_params(labelsize = 15)	
			#plt.colorbar(tSNE_2D, label="Reward")
			fname = "./record/tSNE2D/{}_{}_tSNE2D.png".format(method, i)
			fig_2D.savefig(fname, format = "png")

def tsne2D_fromfile(obs_file_list, reward_file_list, has_interval=False, interval=1000):
	print(f"Reading CSV file...")
	all_obs_list = list()
	all_reward_list = list()
	for obs_file, reward_file in zip(obs_file_list, reward_file_list):
		obs_list = pandas.read_csv(obs_file).values.tolist()
		reward_list = pandas.read_csv(reward_file).values.tolist()
		all_obs_list = all_obs_list + obs_list
		all_reward_list = all_reward_list + reward_list

	tsne2D(all_obs_list, all_reward_list, method="All_agents", has_interval=has_interval, interval=interval)
