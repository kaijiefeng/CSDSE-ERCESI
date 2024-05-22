class config_global():
	def __init__(self, is_setup = False):
		self.MODEL = ["VGG16","MobileNetV2","Mnasnet","ResNet50","Transformer","GNMT"]
		self.CST = ["cloud","largeedge","smalledge"]
		self.period = 1000
		self.MODEL_NUM = len(self.MODEL)
		if(not is_setup):
			self.CST_NUM = len(self.CST)
			self.SCEN_TYPE = self.MODEL_NUM * self.CST_NUM
			self.SCEN_NUM = 1
			self.PROCESS_NUM = 14
		else:
			self.CST_NUM = 1 # setup only in cloud constraint scenario
			self.SCEN_TYPE = self.MODEL_NUM * self.CST_NUM
			self.SCEN_NUM = 1	
			self.PROCESS_NUM = self.MODEL_NUM

		self.TEST_BOUND = int(self.SCEN_NUM * self.MODEL_NUM * self.CST_NUM)
		#### MODEL:{0="VGG16",1="MobileNetV2",2="Mnasnet",3="ResNet50",4="Transformer",5="GNMT"}
		#### CST:{0="cloud",1="largeedge",2="smalledge"}
		PASS_MODEL = [1,2,3,4,5]
		PASS_CST = [1,2]
		self.PASS = list()
		for i_PASS in range(0, self.TEST_BOUND):
			atype = int(i_PASS/self.SCEN_NUM)
			target_type = int(atype/self.MODEL_NUM)
			model_type = atype%self.MODEL_NUM
			if((model_type in PASS_MODEL) or (target_type in PASS_CST)): self.PASS.append(i_PASS)

		self.metrics_name = [
			"latency", # unit: cycle
			"energy", # unit: nJ
			"area", # unit: um^2
			"power", # unit: mW
			"cnt_pes", # unit: /
			"l1_mem", # unit: Byte
			"l2_mem", # unit: Byte
			"edp", # unit: cycle*nJ
			]		

		#### step3 define goal
		self.goal = "latency"
		#self.goal = "energy"
		#self.goal = "edp"
		self.goal_index = self.metrics_name.index(self.goal)
