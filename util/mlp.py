import torch

class mlp_qfunction(torch.nn.Module):
	def __init__(self, space_lenth):
		super(mlp_qfunction, self).__init__()
		self.space_lenth = space_lenth
		self.fc = torch.nn.Linear(256, 128)
		self.fc1 = torch.nn.Linear(self.space_lenth+1, 256)
		self.fc2 = torch.nn.Linear(128, 64)
		self.fc3 = torch.nn.Linear(64, 1)

	def forward(self, qfunction_input):
		out1 = torch.nn.functional.relu(self.fc1(qfunction_input))
		out1 = torch.nn.functional.relu(self.fc(out1))
		out2 = torch.nn.functional.relu(self.fc2(out1))
		return self.fc3(out2)

class mlp_policyfunction(torch.nn.Module):
	def __init__(self, space_lenth, action_scale_list):
		super(mlp_policyfunction, self).__init__()
		self.space_lenth = space_lenth
		self.action_scale_list = action_scale_list
		self.fc1 = torch.nn.Linear(self.space_lenth + 1, 128)
		self.fc2 = torch.nn.Linear(128, 64)
		#layer fc3 is a multi-output mlp
		self.fc3 = list()
		for action_scale in self.action_scale_list:
			self.fc3.append(torch.nn.Linear(64, action_scale))
		self.fc3 = torch.nn.ModuleList(self.fc3)
	
	def forward(self, qfunction_input, dimension_index):
		norm_dimension_index = dimension_index / len(self.action_scale_list)
		input = torch.cat((qfunction_input, torch.tensor(norm_dimension_index).float().view(1)), dim = -1)
		out1 = torch.nn.functional.relu(self.fc1(input))
		out2 = torch.nn.functional.relu(self.fc2(out1))
		out3 = self.fc3[dimension_index](out2)
		return torch.nn.functional.softmax(out3, dim = -1)

class mlp_fillter(torch.nn.Module):
	def __init__(self, space_lenth):
		super(mlp_fillter, self).__init__()
		self.space_lenth = space_lenth
		self.fc1 = torch.nn.Linear(self.space_lenth, 256)
		self.bn1 = torch.nn.BatchNorm1d(256)
		self.fc2 = torch.nn.Linear(256, 128)
		self.bn2 = torch.nn.BatchNorm1d(128)
		self.fc3 = torch.nn.Linear(128, 128)
		self.bn3 = torch.nn.BatchNorm1d(128)
		self.fc4 = torch.nn.Linear(128, 128)
		self.bn4 = torch.nn.BatchNorm1d(128)
		self.fc5 = torch.nn.Linear(128, 64)
		self.bn5 = torch.nn.BatchNorm1d(64)
		self.fc6 = torch.nn.Linear(64, 2)
		self.dropout = torch.nn.Dropout(0.2)

	def forward(self, input):
		x = self.dropout(torch.nn.functional.relu(self.bn1(self.fc1(input))))
		x = self.dropout(torch.nn.functional.relu(self.bn2(self.fc2(x))))
		x = self.dropout(torch.nn.functional.relu(self.bn3(self.fc3(x))))
		x = self.dropout(torch.nn.functional.relu(self.bn4(self.fc4(x))))
		x = self.dropout(torch.nn.functional.relu(self.bn5(self.fc5(x))))
		x = self.fc6(x)
		return torch.nn.functional.softmax(x, dim = -1)


class DDPG_mlp_qfunction(torch.nn.Module):
	def __init__(self, space_lenth, action_scale_list):
		super(DDPG_mlp_qfunction, self).__init__()
		self.space_lenth = space_lenth
		self.action_scale_list = action_scale_list

		self.fc1 = list()
		for action_scale in self.action_scale_list:
			self.fc1.append(torch.nn.Linear(self.space_lenth + action_scale + 1, 256))
		self.fc1 = torch.nn.ModuleList(self.fc1)
		self.fc2 = torch.nn.Linear(256, 128)
		self.fc3 = torch.nn.Linear(128, 1)

	def forward(self, status_input, action_input, dimension_index):
		#pdb.set_trace()
		norm_dimension_index = torch.tensor(dimension_index / len(self.action_scale_list)).float().view(1)
		input0 = torch.cat([status_input, action_input, norm_dimension_index], dim = -1)
		out1 = torch.nn.functional.relu(self.fc1[dimension_index](input0))
		out2 = torch.nn.functional.relu(self.fc2(out1))
		return self.fc3(out2)

class DDPG_mlp_policyfunction(torch.nn.Module):
	def __init__(self, space_lenth, action_scale_list):
		super(DDPG_mlp_policyfunction, self).__init__()
		self.space_lenth = space_lenth
		self.action_scale_list = action_scale_list

		self.fc1 = torch.nn.Linear(self.space_lenth + 1, 128)
		self.fc2 = torch.nn.Linear(128, 64)
		#layer fc3 is a multi-output mlp
		self.fc3 = list()
		for action_scale in self.action_scale_list:
			self.fc3.append(torch.nn.Linear(64, action_scale))
		self.fc3 = torch.nn.ModuleList(self.fc3)
	
	def forward(self, status_input, dimension_index):
		self.max_action = self.action_scale_list[dimension_index] - 1

		norm_dimension_index = torch.tensor(dimension_index / len(self.action_scale_list)).float().view(1)
		input0 = torch.cat([status_input, norm_dimension_index], dim = -1)
		out1 = torch.nn.functional.relu(self.fc1(input0))
		out2 = torch.nn.functional.relu(self.fc2(out1))
		out3 = self.fc3[dimension_index](out2)
		return torch.tanh(out3) #* self.max_action
		#return out3

class rnn_policyfunction(torch.nn.Module):
	def __init__(self, input_lenth, action_scale_list):
		super(rnn_policyfunction, self).__init__()
		self.input_lenth = input_lenth
		self.action_scale_list = action_scale_list
		self.lenth = len(self.action_scale_list)

		self.rnn = torch.nn.LSTM(
			input_size = self.input_lenth+1,
			hidden_size = 128,
			num_layers = 1,
			batch_first = True
		)
		self.fc1 = torch.nn.Linear(128, 64)
		self.fc2 = list()
		for action_scale in self.action_scale_list:
			self.fc2.append(torch.nn.Linear(64, action_scale))
		self.fc2 = torch.nn.ModuleList(self.fc2)

	def forward(self, dimension_index, status_input, rnn_state):
		#### input/output_size:(batch=1, time_step=1, input_size=self.input_lenth+1)
		norm_dimension_index = dimension_index/self.lenth
		input = torch.cat((status_input, torch.tensor(norm_dimension_index).float().view(1)), dim = -1)
		out_rnn, rnn_state = self.rnn(input.view(1, 1, self.input_lenth+1), rnn_state)
		out2 = torch.nn.functional.relu(self.fc1(out_rnn[0,0,:]))
		out3 = self.fc2[dimension_index](out2)
		return torch.nn.functional.softmax(out3, dim = -1), rnn_state
