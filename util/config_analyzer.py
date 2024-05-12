import pandas
import os
import pdb
import sys

from constraints import constraint, constraints
sys.path.append("../")
from config import config_global

class config_self():
	def __init__(self, iindex = 0, is_setup = False):
		global_config = config_global()
		self.SCEN_NUM = global_config.SCEN_NUM
		self.MODEL_NUM = global_config.MODEL_NUM
		self.period = global_config.period
		self.metrics_name = global_config.metrics_name
		self.goal = global_config.goal
		self.goal_index = global_config.goal_index

		#### define model
		atype = int(iindex/self.SCEN_NUM)

		target_type = int(atype/self.MODEL_NUM)
		if(target_type == 0): self.target = "cloud"
		elif(target_type == 1): self.target = "largeedge"
		elif(target_type == 2): self.target = "smalledge"
		else: pass

		model_type = atype%self.MODEL_NUM
		if(model_type == 0): self.nnmodel = "VGG16"
		elif(model_type == 1): self.nnmodel = "MobileNetV2"
		elif(model_type == 2): self.nnmodel = "MnasNet"
		elif(model_type == 3): self.nnmodel = "ResNet50"
		elif(model_type == 4): self.nnmodel = "Transformer"
		elif(model_type == 5): self.nnmodel = "GNMT"
		else: pass

		#### define the constrain

		if(self.target == "cloud"):#cloudTPU
			self.NUMPES_THRESHOLD = 65536
			self.L1SIZE_THRESHOLD = 4000000
			self.L2SIZE_THRESHOLD = 24000000
			self.AREA_THRESHOLD = 331000000
			self.POWER_THRESHOLD = 40000
		elif(self.target == "largeedge"):#NVDLA-LARGE
			self.NUMPES_THRESHOLD = 1024
			self.L1SIZE_THRESHOLD = 27648
			self.L2SIZE_THRESHOLD = 512000
			self.AREA_THRESHOLD = 16000000 
			self.POWER_THRESHOLD = 450 
		elif(self.target == "smalledge"):#eyriess
			self.NUMPES_THRESHOLD = 168
			self.L1SIZE_THRESHOLD = 514
			self.L2SIZE_THRESHOLD = 108000
			self.AREA_THRESHOLD = 16000000
			self.POWER_THRESHOLD = 450

		self.THRESHOLD_RATIO = 2
		cnt_pes = constraint(name = "cnt_pes", threshold = self.NUMPES_THRESHOLD, threshold_ratio = self.THRESHOLD_RATIO)
		l1_mem = constraint(name = "l1_mem", threshold = self.L1SIZE_THRESHOLD, threshold_ratio = self.THRESHOLD_RATIO)
		l2_mem = constraint(name = "l2_mem", threshold = self.L2SIZE_THRESHOLD, threshold_ratio = self.THRESHOLD_RATIO)				
		area = constraint(name = "area", threshold = self.AREA_THRESHOLD, threshold_ratio = self.THRESHOLD_RATIO)
		power = constraint(name = "power", threshold = self.POWER_THRESHOLD, threshold_ratio = self.THRESHOLD_RATIO)
		self.constraints = constraints()
		self.constraints.append(cnt_pes)
		self.constraints.append(l1_mem)
		self.constraints.append(l2_mem)
		self.constraints.append(area)
		self.constraints.append(power)

		self.is_adaptive = True
		self.is_const = False

		if(not is_setup):
			baseline_filepath = "./data/baseline_{}.csv".format(self.nnmodel)
			assert(os.path.exists(baseline_filepath))
			self.baseline = self.load_metric_baseline(baseline_filepath)
			self.baseline_max = self.load_metric_baseline_max(baseline_filepath)
			self.goal_baseline = self.baseline[self.goal]

	def load_metric_baseline(self, filepath):
		#### get the max value in initial_buffer as the baseline
		dataframe_baseline = pandas.read_csv(filepath)
		data_baseline = dataframe_baseline[self.metrics_name]
		baseline = data_baseline.to_dict(orient ="records")[0]

		#### get the average value in initial_buffer as the baseline
		metrics_avg_name = list()
		for name in self.metrics_name:
			metrics_avg_name.append(name + "_avg")
		data_average = dataframe_baseline[metrics_avg_name]
		baseline_avg = data_average.to_dict(orient ="records")[0]
		for key, avg_key in zip( list(baseline.keys()), list(baseline_avg.keys()) ):
			baseline[key] = baseline_avg[avg_key]

		return baseline

	def load_metric_baseline_max(self, filepath):
		#### get the max value in initial_buffer as the baseline
		dataframe_baseline = pandas.read_csv(filepath)
		data_baseline = dataframe_baseline[self.metrics_name]
		baseline = data_baseline.to_dict(orient ="records")[0]

		return baseline

	def config_check(self):
		print(f"######Config Check######")
		print(f"configtype:test")
		print(f"nnmodel:{self.nnmodel}")
		print(f"target:{self.target}")
		for constraint in self.constraints.constraint_list:
			print(f"{constraint.get_name()}:{constraint.get_threshold()}")
		print(f"goal:{self.goal}")
