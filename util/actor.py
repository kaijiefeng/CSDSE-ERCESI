from space import dimension_discrete
from space import design_space

import random
import math
import numpy
import torch
import pdb

def status_normalize(status, design_space):
	status_copy = dict()
	baseline = design_space.upbound
	for key in status.keys():
		status_copy[key] = status[key]/baseline[key]
	return status_copy

def action_value_normalize(action_list, design_space):
	action_list_copy = list()
	for index, item in enumerate(design_space.dimension_box):
		action_list_copy.append(action_list[index] / len(item.sample_box))
	return action_list_copy

def compute_action_distance(action_list_a, action_list_b, design_space):
	action_list_a_normalized = action_value_normalize(action_list_a, design_space)
	action_list_b_normalized = action_value_normalize(action_list_b, design_space)
	distance = 0
	for i, j in zip(action_list_a_normalized, action_list_b_normalized):
		distance = distance + (i-j)**2
	return distance

def action_normalize(action_tensor, design_space, step):
	action_tensor = action_tensor / (design_space.get_dimension_scale(step)-1)

def status_to_tensor(status):
	_list = []
	#### step1:get list
	for index in status:
		_list.append(status[index])
	#### step2:get numpy_array
	_ndarray = numpy.array(_list)
	#### step3:get tensor
	_tensor = torch.from_numpy(_ndarray)
	return _tensor

def status_to_Variable(status):
	_list = []
	#### step1:get list
	for index in status:
		_list.append(status[index])
	#### step2:get numpy_array
	_ndarray = numpy.array(_list)
	#### step3:get tensor
	_tensor = torch.from_numpy(_ndarray)
	_Variable = torch.autograd.Variable(_tensor).float()
	return _Variable	

def status_to_list(status):
	_list = []
	#### step1:get list
	for index in status:
		_list.append(status[index])
	return _list

def index_to_one_hot(scale, action_index):
	_tensor = torch.zeros(scale)
	_tensor.scatter_(dim = 0, index = torch.as_tensor(action_index), value = 1)
	return _tensor

def log_density(x, mu, std, logstd):
    var = std.pow(2)
    log_density = -(x - mu).pow(2) / (2 * var) \
                  - 0.5 * math.log(2 * math.pi) - logstd
    return log_density.sum(1, keepdim=True)

def get_action(mu, std):
    action = torch.normal(mu, std)
    action = action.data.numpy()
    return action

def normal_density(x, mean, sigma):
	return 1/((2 * 3.1415)**0.5 * sigma) \
		   * math.exp(- (x - mean)**2/(2 * sigma**2))

def get_normal_tensor(design_space, action_index, dimension_index, model_sigma):
	sample_box = design_space.get_dimension_sample_box(dimension_index)
	normal_list = []
	for sample_index, value in enumerate(sample_box):
		normal_list.append(normal_density(sample_index, action_index, model_sigma))
	normal_tensor = torch.from_numpy(numpy.array(normal_list))
	normal_tensor = normal_tensor / normal_tensor.sum()
	return normal_tensor

def get_log_prob(policyfunction, design_space, status, action_index, dimension_index):
	status_normalization = status_normalize(status, design_space)
	probs = policyfunction(status_to_Variable(status_normalization), dimension_index)
	#### compute entropy
	entropy = -(probs * probs.log()).sum()
	#### use multinomial to realize the sampling of policy function
	action_index_tensor = index_to_one_hot(len(probs), action_index)
	#### use onehot index to restore compute graph
	prob_sampled = (probs * action_index_tensor).sum()
	log_prob_sampled = prob_sampled.log()

	return entropy, log_prob_sampled

def get_kldivloss_and_log_prob(policyfunction, design_space, status, action_index, dimension_index):
	status_normalization = status_normalize(status, design_space)
	probs = policyfunction(status_to_Variable(status_normalization), dimension_index)
	#### compute entropy
	entropy = -(probs * probs.log()).sum()
	#### compute kl div between probs and target distribution
	model = design_space.get_dimension_model(dimension_index)
	if(model["name"] == "normal"):
		target_distribution = get_normal_tensor(design_space, action_index, dimension_index, model["param"]).float()
		#if(dimension_index == 2): print(f"normal: target_distribution{target_distribution}")	
	elif(model["name"] == "one_hot"):
		target_distribution = index_to_one_hot(len(probs), action_index)
	kldivloss = torch.nn.functional.kl_div(probs.log(), target_distribution, reduction = "sum")
	#### use multinomial to realize the sampling of policy function
	action_index_tensor = index_to_one_hot(len(probs), action_index)
	#### use onehot index to restore compute graph
	prob_sampled = (probs * action_index_tensor).sum()
	log_prob_sampled = prob_sampled.log()

	return entropy, kldivloss, log_prob_sampled

def csdse_get_log_prob(policyfunction, obs, action_index, dimension_index):
	obs = torch.autograd.Variable(torch.tensor(obs)).float()
	probs = policyfunction(obs, dimension_index)
	#### compute entropy
	entropy = -(probs * probs.log()).sum()
	#### use multinomial to realize the sampling of policy function
	action_index_tensor = index_to_one_hot(len(probs), action_index)
	#### use onehot index to restore compute graph
	prob_sampled = (probs * action_index_tensor).sum()
	log_prob_sampled = prob_sampled.log()

	return entropy, log_prob_sampled

def get_log_prob_rnn(policyfunction, obs, action_index, dimension_index, rnn_state_train):
	obs = torch.autograd.Variable(torch.tensor(obs)).float().view(1)
	probs, rnn_state_train = policyfunction(dimension_index, obs, rnn_state_train)
	#### compute entropy
	entropy = -(probs * probs.log()).sum()
	#### use multinomial to realize the sampling of policy function
	action_index_tensor = index_to_one_hot(len(probs), action_index)
	#### use onehot index to restore compute graph
	prob_sampled = (probs * action_index_tensor).sum()
	log_prob_sampled = prob_sampled.log()

	return entropy, log_prob_sampled, rnn_state_train

def get_kldivloss_and_log_prob_rnn(policyfunction, design_space, status, action_index, dimension_index, obs, rnn_state_train):
	obs = torch.autograd.Variable(torch.tensor(obs)).float().view(1)
	probs, rnn_state_train = policyfunction(dimension_index, obs, rnn_state_train)
	#### compute entropy
	entropy = -(probs * probs.log()).sum()
	#### compute kl div between probs and target distribution
	model = design_space.get_dimension_model(dimension_index)
	if(model["name"] == "normal"):
		target_distribution = get_normal_tensor(design_space, action_index, dimension_index, model["param"]).float()
		#if(dimension_index == 2): print(f"normal: target_distribution{target_distribution}")	
	elif(model["name"] == "one_hot"):
		target_distribution = index_to_one_hot(len(probs), action_index)
	kldivloss = torch.nn.functional.kl_div(probs.log(), target_distribution, reduction = "sum")
	#### use multinomial to realize the sampling of policy function
	action_index_tensor = index_to_one_hot(len(probs), action_index)
	#### use onehot index to restore compute graph
	prob_sampled = (probs * action_index_tensor).sum()
	log_prob_sampled = prob_sampled.log()

	return entropy, kldivloss, log_prob_sampled

class actor_random():
        #### make_policy return the index of dimension's sample_box rather than the actual value for convinience
	def make_policy(self, design_space, dimension_index):
		return random.randint(0,design_space.dimension_box[dimension_index].get_scale()-1)	

class actor_e_greedy():
	def __init__(self):
		self.greedy_possiblity = 0.7
		self.sample_range = int(2)
		
	def action_choose(self, qfunction, design_space, dimension_index, ratio):
		#### constrain the sample range
		#### python range is [a,b), so the up bound requre a +1
		current_index = design_space.get_dimension_current_index(dimension_index)
		sample_bottom = max(0, current_index - self.sample_range)
		sample_top = min(int(design_space.get_dimension_scale(dimension_index)), current_index + self.sample_range + 1)
		
		if(random.random() < self.greedy_possiblity):
		#### greedy search best action
			self.best_action_index = 0
			self.best_qvalue = 0
			#### find the best action in that dimension
			for action_index in range(int((design_space.get_dimension_scale(dimension_index)))):
			#for action_index in range(sample_bottom, sample_top):
				design_space.sample_one_dimension(dimension_index, action_index)
				status = design_space.get_compact_status(dimension_index)
				with torch.no_grad():
					#### compute the q value
					step = (dimension_index+1) / design_space.get_lenth()
					step = torch.tensor(step).float().view(1)
					status = status_normalize(status, design_space)
					variable = status_to_Variable(status)
					variable = torch.cat((variable, step), dim = -1)

					qvalue = qfunction(variable)
				##### compare and find the best q value
				if(qvalue > self.best_qvalue):
					self.best_action_index = action_index
					self.best_qvalue = qvalue
		else:
		#### random choose an action
			self.best_action_index = random.randint(0, design_space.get_dimension_scale(dimension_index) - 1)
			#self.best_action_index = random.randint(sample_bottom, sample_top - 1)
		return self.best_action_index

	'''
	def ratiochange_action_choose(self, qfunction, design_space, dimension_index, ratio):
		#### constrain the sample range
		#### python range is [a,b), so the up bound requre a +1
		current_index = design_space.get_dimension_current_index(dimension_index)
		sample_bottom = max(0, current_index - self.sample_range)
		sample_top = min(int(design_space.get_dimension_scale(dimension_index)), current_index + self.sample_range + 1)
		
		if(random.random() < self.greedy_possiblity**ratio):
		#### greedy search best action
			self.best_action_index = 0
			self.best_qvalue = 0
			#### find the best action in that dimension
			for action_index in range(int((design_space.get_dimension_scale(dimension_index)))):
			#for action_index in range(sample_bottom, sample_top):
				status = design_space.sample_one_dimension(dimension_index, action_index)
				with torch.no_grad():
					#### compute the q value
					step = dimension_index / design_space.get_lenth()
					step = torch.tensor(step).float().view(1)
					status = status_normalize(status, design_space)
					variable = status_to_Variable(status)
					variable = torch.cat((variable, step), dim = -1)

					qvalue = qfunction(variable)
				##### compare and find the best q value
				if(qvalue > self.best_qvalue):
					self.best_action_index = action_index
					self.best_qvalue = qvalue
		else:
		#### random choose an action
			self.best_action_index = random.randint(0, design_space.get_dimension_scale(dimension_index) - 1)
			#self.best_action_index = random.randint(sample_bottom, sample_top - 1)
		return self.best_action_index
	'''
	
	def random_action_choose(self, qfunction, design_space, dimension_index, ratio):
		#### constrain the sample range
		#### python range is [a,b), so the up bound requre a +1
		current_index = design_space.get_dimension_current_index(dimension_index)
		sample_bottom = max(0, current_index - self.sample_range)
		sample_top = min(int(design_space.get_dimension_scale(dimension_index)), current_index + self.sample_range + 1)		

		#### random choose an action
		self.best_action_index = random.randint(0, design_space.get_dimension_scale(dimension_index) - 1)
		#self.best_action_index = random.randint(sample_bottom, sample_top - 1)
		return self.best_action_index

	def best_action_choose(self, qfunction, design_space, dimension_index):
		##### greedy search best action
		self.true_best_action_index = 0
		self.true_best_qvalue = 0
		##### find the best action in that dimension
		for action_index in range(int((design_space.get_dimension_scale(dimension_index)))):
			status = design_space.sample_one_dimension(dimension_index, action_index)
			with torch.no_grad():
				step = (dimension_index+1) / design_space.get_lenth()
				step = torch.tensor(step).float().view(1)
				status = status_normalize(status, design_space)
				variable = status_to_Variable(status)
				variable = torch.cat((variable, step), dim = -1)
				
				qvalue = qfunction(variable)
			##### compare and find the best q value
			if(qvalue > self.true_best_qvalue):
				self.true_best_action_index = action_index
				self.true_best_qvalue = qvalue
		return self.true_best_action_index

class actor_policyfunction():
	def action_choose(self, policyfunction, design_space, status, dimension_index):
		status_normalization = status_normalize(status, design_space)
		probs = policyfunction(status_to_Variable(status_normalization), dimension_index)
		use_noise = False
		if(use_noise):		
			noise = torch.normal(mean = torch.zeros_like(probs), std = 0.005)
			probs_noise = probs + noise
			probs_noise = torch.clamp(probs_noise, 0, 1)
			action_index = probs_noise.multinomial(num_samples = 1).data
		else:
			action_index = probs.multinomial(num_samples = 1).data
		#### compute entropy
		entropy = -(probs * probs.log()).sum()
		#### use multinomial to realize the sampling of policy function
		action_index_tensor = index_to_one_hot(len(probs), action_index)
		#### use onehot index to restore compute graph
		prob_sampled = (probs * action_index_tensor).sum()
		log_prob_sampled = prob_sampled.log()

		#if(dimension_index == 8):
		#	print(f"\nstep:{dimension_index}, probs:{probs}")
		return entropy, action_index, log_prob_sampled

	def action_choose_with_no_grad(self, policyfunction, design_space, status, dimension_index, std = 0.01, is_train = True):
		with torch.no_grad():
			dimension = design_space.dimension_box[dimension_index]
			if(dimension.get_scale() != 1):
				status_normalization = status_normalize(status, design_space)
				probs = policyfunction(status_to_Variable(status_normalization), dimension_index)
				if(is_train):
					noise = torch.normal(mean = torch.zeros_like(probs), std = std)
					probs_noise = probs + noise
					probs_noise = probs_noise + torch.abs(probs_noise.min())
					probs_noise = probs_noise / torch.sum(probs_noise, dim = -1)
					#print(f"probs_noise:{probs_noise}")
				else:
					probs_noise = probs
				action_index = probs_noise.multinomial(num_samples = 1).data
			else:
				action_index = torch.tensor(dimension.get_current_index())
		return action_index

	'''
	def sldse_action_choose_with_no_grad(self, policyfunction, tutor, design_space, dimension_index, std = 0.1, w_probs = 0.9, is_train = True):
		status = design_space.get_status()
		with torch.no_grad():
			status_normalization = status_normalize(status, design_space)
			probs = policyfunction(status_to_Variable(status_normalization), dimension_index)
			trend = tutor(status_to_Variable(status_normalization), dimension_index)
			w_probs = w_probs
			w_trend = 1 - w_probs

			if(is_train):
				#noise = torch.normal(mean = w_trend * trend, std = std)
				#noise = torch.normal(mean = torch.zeros_like(probs), std = std)
				probs_noise = torch.normal(mean = probs, std = std)
				#probs_noise = torch.normal(mean = w_probs * probs + w_trend * trend, std = std)
				probs_noise = torch.clamp(probs_noise, 0, 1)
				probs_noise = probs_noise / torch.sum(probs_noise, dim = -1)
			else:
				probs_noise = probs
			#### use multinomial to realize the sampling of policy function
			action_index = probs_noise.multinomial(num_samples = 1).data

		return action_index, probs_noise[action_index].data
	'''

	def action_choose_DDPG(self, policyfunction, design_space, status, dimension_index):
		with torch.no_grad():
			status = status_normalize(status, design_space)
			probs = policyfunction(status_to_Variable(status), dimension_index)
			probs_softmax = torch.softmax(probs, dim = -1)
			#### use multinomial to realize the sampling of policy function
			action_index = probs_softmax.multinomial(num_samples = 1).data
			#print(f"probs:{probs}")
			#action_index = torch.argmax(probs)
		return action_index, probs

	def action_choose_rnn(self, policyfunction, design_space, status, dimension_index, rnn_state, std = 0.01, is_train = True):
		with torch.no_grad():
			dimension = design_space.dimension_box[dimension_index]
			if(dimension.get_scale() != 1):
				status_normalization = status_normalize(status, design_space)
				probs, rnn_state = policyfunction(dimension_index, status_to_Variable(status_normalization), rnn_state)
				#print(f"index:{dimension_index}, probs:{probs}")
				if(is_train):
					noise = torch.normal(mean = torch.zeros_like(probs), std = std)
					probs_noise = probs + noise
					probs_noise = probs_noise + torch.abs(probs_noise.min())
					probs_noise = probs_noise / torch.sum(probs_noise, dim = -1)
					#print(f"probs_noise:{probs_noise}")
				else:
					probs_noise = probs
				action_index = probs_noise.multinomial(num_samples = 1).data
			else:
				action_index = torch.tensor(dimension.get_current_index())
		return action_index, rnn_state