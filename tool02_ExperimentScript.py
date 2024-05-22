import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

proms = [
#	"test01_RGS_MAESTRO.py",
#	"test02_GA_MAESTRO.py",
#	"test02_1_NSGA2_MAESTRO.py",
#	"test03_BO_MAESTRO.py",
#	"test04_RL_DQN_MAESTRO.py",
#	"test05_RL_REINFORCE.py",
#	"test06_RL_DDPG_MAESTRO.py",
#	"test07_RL_PPO_MAESTRO.py",
#	"test08_RL_SAC_MAESTRO.py",
#	"test09_RL_ERDSE_MAESTRO.py",
#	"test10_RL_ACDSE_MAESTRO.py",
	"test11_RL_CSDSE_MAESTRO.py"
#   "test12_RL_HAPPO_MAESTRO.py"
]

for prom in proms:
	os.system("python3 {}".format(prom))