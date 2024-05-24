# **CSDSE**

***

CSDSE (++C++ooperative ++S++earch based ++D++esign ++S++pace ++E++xploration method) is an efficient design space exploration (DSE) method for DNN accelerator. Based on Reinforcement Learning and Cooperative Search Mechanism, CSDSE enables agile co-design of accelerator architecture and dataflow mapping. This repository contains the detailed implementations of CSDSE and related baselines, including ERDSE, ACDSE and other DSE methods based on Genetic Algorithm (GA), Bayesian Optimization (BO) and Reinforcement Learning (RL), providing a integrated environment for comparing various DSE algorithms and methods.

### **Installation**

***

#### **Install CSDSE**

```bash
conda create -n csdse python=3.7
conda activate csdse
mkdir DSE
cd DSE
git clone https://github.com/kaijiefeng/CSDSE-ERCESI.git
```

#### **Dependencies**

1\.	Install pytorch, matplotli and pandas

```bash
pip install torch==1.13.1
pip install matplotlib==3.5.1
pip install pandas==1.3.5
```

(Optional) Install geatpy if you want to evaluate DSE methods based on GA (corresponding implementations are located in file test02\_GA\_MAESTRO.py).&#x20;

```bash
pip install geatpy==2.5.1
```

(Optional) Install hyperopt if you want to evaluate DSE methods based on BO(corresponding implementations are located in file `test03_BO_MAESTRO.py`).

```bash
pip install hyperopt==0.2.7
```

(Optional) You can also run following command to install dependencies at once:

```bash
pip install -r requirement.txt
```

2\.	Install MAESTRO for accelerator performance evaluation (if the installation is stuck for any issue, please refer to the official installation procedure of MAESTRO in this [link](https://maestro.ece.gatech.edu/docs/build/html/installation.html)):

```bash
# please notice that the repository of MAESTRO should be located at the same directory of CSDSE
cd DSE
git clone https://github.com/maestro-project/maestro
cd maestro
# if your environment is not implemented with Boost library, please first run following command; else you can skip this step
sudo apt-get install libboost-all-dev
# after installing Boost library, run following command:
scons
```

### **Usage**

***

#### **Conduct DSE via Running Script**

You can conveniently conduct DSE via running script `tool02_ExperimentScript.py`:

```bash
cd DSE
cd CSDSE-ERCESI
python tool02_ExperimentScript.py
```

In this script, you can select to conduct one or more DSE methods once via disable unnecessary options (in default, only CSDSE in `test11_RL_CSDSE_MAESTRO.py` is enabled):

```python
import os 
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

proms = [
#	"test01_RGS_MAESTRO.py", #### Random Search
#	"test02_GA_MAESTRO.py", #### Standard Genetic Algorithm (GA)
#	"test02_1_NSGA2_MAESTRO.py", #### Non-dominated Sorting Genetic Algorithm II (NSGA-II), for multi-objective optimization
#	"test03_BO_MAESTRO.py", #### Bayesian Optimization (BA)
#	"test04_RL_DQN_MAESTRO.py", #### RL algorithm: Deep Q-Network (DQN), not avaliable in current version
#	"test05_RL_REINFORCE.py", #### RL algorithm: REINFORCE
#	"test06_RL_DDPG_MAESTRO.py", #### RL algorithm: Deep Deterministic Policy Gradient (DDPG), not avaliable in current version
#	"test07_RL_PPO_MAESTRO.py", #### RL algorithm: Proximal Policy Optimization (PPO)
#	"test08_RL_SAC_MAESTRO.py", #### RL algorithm: Soft Actor-Critic (SAC)
#	"test09_RL_ERDSE_MAESTRO.py", #### RL algorithm: ERDSE, baseline of CSDSE, SOTA work in DNN accelrator DSE
#	"test10_RL_ACDSE_MAESTRO.py", #### RL algorithm: ACDSE, baseline of CSDSE, SOTA work in DNN accelrator DSE
	"test11_RL_CSDSE_MAESTRO.py" #### RL algorithm: CSDSE
#   "test12_RL_HAPPO_MAESTRO.py" #### RL algorithm: HAPPO, not avaliable in current version
]

for prom in proms:
	os.system("python3 {}".format(prom))
```

#### **Experiment Setting Configuration**&#x20;

You can edit file `config.py` to adjust the experiment setting:

```python
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
```

In default, it will conduct DSE in the scenario of (model=VGG16, constraint=Cloud, objective=latency), You can change the model, constraint and objective via defining parameter `PASS_MODEL`, `PASS_CST`, `self.goal` respectively. Models and constraints involved in `PASS_MODEL`, `PASS_CST` will NOT be evaluated during DSE. Thereby, you can disable unnecessary models and constraints via inserting their indexes in lists of `PASS_MODEL`, `PASS_CST`. Objective choices include latency, energy and energy-delay-product (EDP), you can define `self.goal = "latency"`, `self.goal = "energy"` or `self.goal = "edp"` to alter the objective.

Meanwhile, you can define `self.SCEN_NUM` and `self.PROCESS_NUM` to repeatedly conduct the DSE in specific scenarios on multiple parallel progresses, which is employed to evaluated the stability of DSE methods. `self.SCEN_NUM` defines the repeat count (with different seeds) and `self.PROCESS_NUM` defines the progress count.

#### **CSDSE Configuration**

When you implement CSDSE to conduct the DSE, you can define the `member_list` to adjust the agent division. In default, CSDSE is implemented with 8 heterogeneous agents. &#x9;

```python
member_list = ["brave_phd", "discreet_phd", "brave_master", "discreet_master", "brave_tutor", "discreet_tutor", "brave_reserve", "discreet_reserve"]
```

#### **Records of Optimization Results**

The optimization results of CSDSE will be output in directory `/DSE/CSDSE-ERCESI/record/objectvalue/<Alogorithm_Name>_<Objective>.csv`, listing per-epoch normalized  optimization results of all scenarios. It should be noticed that each scenario will be evaluated for multiple times, defined by `self.SCEN_NUM` in `config.py`, and optimization results of each evaluation will be recorded. Meanwhile, the average results of each scenarios will also be calculated and recorded. Following is an example of records:

| 0        | 1        | 2        | 3        | 4        |
| :------- | :------- | :------- | :------- | :------- |
| 1000     | 0.183539 | 0.212023 | 1000     | 0.180748 |
| 0.267669 | 0.183539 | 0.212023 | 1.142548 | 0.180748 |
| 0.267669 | 0.183539 | 0.212023 | 1.142548 | 0.180748 |
| 0.170031 | 0.183539 | 0.212023 | 1.142548 | 0.180748 |
| 0.170031 | 0.183539 | 0.212023 | 1.142548 | 0.180748 |
| 0.170031 | 0.183539 | 0.212023 | 1.142548 | 0.180748 |
| 0.170031 | 0.183539 | 0.212023 | 0.209222 | 0.180748 |
| 0.170031 | 0.183539 | 0.212023 | 0.209222 | 0.180748 |
| 0.170031 | 0.183539 | 0.212023 | 0.209222 | 0.180748 |
| 0.170031 | 0.164538 | 0.180996 | 0.209222 | 0.180748 |

The evaluation indexes are listed in the first row. In this example, the scenario of (model=VGG16, constraint=Cloud, objective=latency) is selected and the evaluation count `self.SCEN_NUM = 5`. Therefore, the scenario count is 1, the evaluation count is 5 and the evaluation indexes ranges from 0 to 4.

The per-epoch normalized optimization results are listed in row #2 to #11. CSDSE has randomly sampled and evaluated 1000 design points for each model, calculating the average value of each performance metrics as the normalization baselines (listed in `/DSE/CSDSE-ERCESI/data/baseline`*`<`*`Model_Name>.csv`). Therefore, the ground truth value of optimization results should be the products the normalized values and baselines. Following table lists the metric baselines of VGG16:

| latency\_avg/cycle | energy\_avg/nJ | area\_avg/um2 | power\_avg/mw | cnt\_pes\_avg | l1\_mem\_avg/Byte | l2\_mem\_avg/Byte | edp\_avg/cycle\*nJ |
| :----------------- | :------------- | :------------ | :------------ | :------------ | :---------------- | :---------------- | :----------------- |
| 3.0697E+09         | 3.7390E+07     | 1.0385E+11    | 1.1442E+02    | 5.8789E+03    | 5.4000E+01        | 1.2144E+06        | 1.4279E+17         |

