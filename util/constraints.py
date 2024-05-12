class constraint():
	def __init__(self, name, threshold, threshold_ratio):
		self.name = name
		self.threshold = threshold
		self.threshold_ratio = threshold_ratio

		self.value = 0
		self.margin = (self.threshold - self.value)/self.threshold
		self.stable_margin = self.margin
		self.is_meet_flag = False
		self.l = 0
		self.punishment = 0
	def update(self, value):
		self.value = value
		self.margin = (self.threshold - self.value)/self.threshold
		if(self.margin < 0):
			self.is_meet_flag = False
			self.l = self.threshold_ratio
		else:
			self.is_meet_flag = True
			self.l = 0
		self.punishment = (self.value/self.threshold)**self.l
	def update_stable_margin(self):
		self.stable_margin = self.margin
	def get_name(self):
		return self.name	
	def get_threshold(self):
		return self.threshold
	def get_margin(self):
		return self.margin
	def get_stable_margin(self):
		return self.stable_margin
	def is_meet(self):
		return self.is_meet_flag
	def get_punishment(self):
		return self.punishment
	def print(self):
		print(f"name = {self.name}, threshold = {self.threshold}, value = {self.value}, margin = {self.margin}, is_meet:{self.is_meet_flag}, punishment = {self.punishment}, stable_margin = {self.stable_margin}")

class constraints():
	def __init__(self):
		self.constraint_list = list()
	def append(self, constraint):
		self.constraint_list.append(constraint)
	def update(self, value_dict):
		assert(len(value_dict.items()) == len(self.constraint_list))
		for key, value, constrain in zip(value_dict.keys(), value_dict.values(), self.constraint_list):
			assert(key == constrain.get_name())
			constrain.update(value)
	def multi_update(self, metrics):
		for constraint in self.constraint_list:
			name = constraint.get_name()
			constraint.update(metrics[name])
	def update_stable_margin(self):
		for constraint in self.constraint_list:
			constraint.update_stable_margin()
	def get_stable_margin(self, name_str):
		stable_margin = 0
		for constraint in self.constraint_list:
			if(constraint.get_name() == name_str):
				stable_margin = constraint.get_stable_margin()
		return stable_margin
	def is_any_margin_meet(self, ratio):
		is_any_margin_meet_flag = False
		for constraint in self.constraint_list:
			is_any_margin_meet_flag = is_any_margin_meet_flag or (constraint.get_stable_margin() <= ratio)
		return is_any_margin_meet_flag
	def get_threshold(self, name_str):
		threshold = 0
		for constraint in self.constraint_list:
			if(constraint.get_name() == name_str):
				threshold = constraint.get_threshold()
		return threshold
	def is_any_meet(self):
		self.is_any_meet_flag = False
		for constraint in self.constraint_list:
			self.is_any_meet_flag = self.is_any_meet_flag or constraint.is_meet() 
		return self.is_any_meet_flag
	def is_all_meet(self):
		self.is_all_meet_flag = True
		for constraint in self.constraint_list:
			self.is_all_meet_flag = self.is_all_meet_flag and constraint.is_meet()
		return self.is_all_meet_flag
	def get_punishment(self):
		self.punishment = 1
		for constraint in self.constraint_list:
			self.punishment = self.punishment * constraint.get_punishment()
		return self.punishment
	def print(self):
		for constraint in self.constraint_list:
			constraint.print()
	def get_name_list(self):
		constraints_name_list = list()
		for constraint in self.constraint_list:
			constraints_name_list.append(constraint.get_name())
		return constraints_name_list