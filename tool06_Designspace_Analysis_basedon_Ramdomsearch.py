import pdb
import random
import numpy as np
import pandas
import os
import sys
import matplotlib.pyplot as plt
from sklearn import manifold
from multiprocessing import Process, Lock, Manager, Pool

from config import config_global
sys.path.append("./util/")
from space import dimension_discrete, design_space, create_space_maestro
from actor import actor_random
from evaluation_maestro import evaluation_maestro
from config_analyzer import config_self
from timer import timer
from recorder import recorder

os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

def run(args):
	iindex, reward_record, obs_record, best_record = args
	print(f"%%%%TEST{iindex} START%%%%")

	seed = iindex * 10000
	np.random.seed(seed)
	random.seed(seed)

	config = config_self(iindex)
	constraints = config.constraints
	nnmodel = config.nnmodel
	goal = config.goal
	baseline = config.baseline
	target = config.target
	config.config_check()
	pid = os.getpid()

	DSE_action_space = create_space_maestro(nnmodel, target = target)	
	evaluation = evaluation_maestro(iindex, nnmodel, pid, DSE_action_space)
	actor = actor_random()

	reward_list = list()
	norm_reward_list = list()
	obs_list = list()
	best_reward = 0
	t = timer()

	upbound_for_period = config.period
	count_period = 0
	t.start("all")
	while(count_period < upbound_for_period):	
		count_period = count_period + 1

		DSE_action_space.status_reset()
		for step in range(DSE_action_space.get_lenth()):
			DSE_action_space.sample_one_dimension(dimension_index = step, sample_index = actor.make_policy(DSE_action_space, step))

		status = DSE_action_space.get_status()
		t.start("eva")
		metrics = evaluation.evaluate(status)
		t.end("eva")
		if(metrics != None):
			constraints.multi_update(metrics)
			objectvalue = metrics[goal]
			reward = 1 / (objectvalue * constraints.get_punishment())
		else:
			reward = 0
		if(reward > best_reward): best_reward = reward
		reward_list.append(reward)
		obs_list.append(DSE_action_space.get_obs().tolist())
		print(f"period:{count_period}, objectvalue:{objectvalue}, reward:{reward}, best_reward:{best_reward}", end = '\r')

	print(f"best_reward:{best_reward}")
	path1 = "./record/analysis/{}_{}_{}_reward.csv".format(nnmodel, target, goal)
	py_reward_record = np.array(reward_list).T
	reward_df = pandas.DataFrame(py_reward_record)
	reward_df.to_csv(path1, index = None, header = None)

	path2 = "./record/analysis/{}_{}_{}_obs.csv".format(nnmodel, target, goal)
	py_vector_record = np.array(obs_list)
	vector_df = pandas.DataFrame(py_vector_record)
	vector_df.to_csv(path2, index = None, header = None)

	reward_record[nnmodel+target] = reward_list
	obs_record[nnmodel+target] = obs_list
	best_record[nnmodel+target] = best_reward

	print(f"%%%%TEST{iindex} END%%%%")

def analysis(args):
	iindex, reward_record, obs_record, best_record = args
	print(f"%%%%TEST{iindex} START to ANALYSIS%%%%")	
	config = config_self(iindex)
	nnmodel = config.nnmodel
	target = config.target
	goal = config.goal

	best_reward_global = 0
	for scenario in [nnmodel+"cloud", nnmodel+"largeedge", nnmodel+"smalledge"]:
		best_reward = best_record[scenario]
		if(best_reward > best_reward_global): best_reward_global = best_reward

	reward_list = reward_record[nnmodel+target]
	vector_list = obs_record[nnmodel+target]

	norm_reward_list = list()
	for reward in reward_list:
		norm_reward = reward / best_reward_global
		norm_reward_list.append(norm_reward)
	norm_reward_list.sort(reverse = False)

	import matplotlib
	print(f"font:{matplotlib.matplotlib_fname()}")
	action_array = np.array(vector_list)
	reward_continue_array = np.array(norm_reward_list)
	tsne = manifold.TSNE(n_components = 2, init = "pca", random_state = 1)
	print(f"Start t-SNE")
	x_tsne = tsne.fit_transform(action_array)
	x_min, x_max = x_tsne.min(0), x_tsne.max(0)
	x_norm = (x_tsne - x_min)/(x_max - x_min)
	r_max = reward_continue_array.max(0)
	x = x_norm
	r = reward_continue_array

	fig_3D = plt.figure(dpi=1200)
	tSNE_3D = plt.axes(projection = '3d')
	tSNE_3D.scatter3D(x[:, 0], x[:, 1], r, c = r, vmax = 1, cmap = "rainbow", s = 15, alpha = 0.5)
	tSNE_3D.set_xlabel("x")
	tSNE_3D.set_ylabel("y")
	tSNE_3D.set_zlabel("Normalized Reward")
	tSNE_3D.set_xlim((1, 0))
	tSNE_3D.set_ylim((0, 1))
	tSNE_3D.set_zlim((0, 1))
	tSNE_3D.set_zticks([0, 0.2, 0.4, 0.6, 0.8, 1])	
	fname = "./record/analysis/{}_{}_{}_tSNE3D.png".format(nnmodel, target, goal)
	fig_3D.savefig(fname, format = "png")

	# fig_2D = plt.figure(dpi=1200)
	# tSNE_2D = plt.axes()
	# tSNE_2D.scatter(x[:, 0], x[:, 1], c = r, vmax = r_max, cmap = "rainbow", s=15, alpha = 0.5)
	# tSNE_2D.set_xlabel("x")
	# tSNE_2D.set_ylabel("y")
	# fname = "./record/analysis/{}_{}_{}_tSNE2D.png".format(nnmodel, target, goal)
	# fig_2D.savefig(fname, format = "png")
	print(f"%%%%ANALYSIS{iindex} END%%%%")

if __name__ == '__main__':
	algoname = "RGS_Analysis"
	use_multiprocess = True
	global_config = config_global()
	TEST_BOUND = global_config.TEST_BOUND
	PROCESS_NUM = global_config.PROCESS_NUM
	SCEN_TYPE = global_config.SCEN_TYPE
	SCEN_NUM = global_config.SCEN_NUM
	PASS = global_config.PASS

	args_list = list()
	reward_record = Manager().dict()
	obs_record = Manager().dict()
	best_record = Manager().dict()

	if(use_multiprocess):
		args_list = list()
		for iindex in range(TEST_BOUND):
			if(iindex in PASS): continue
			args_list.append((iindex, reward_record, obs_record, best_record))
		pool = Pool(PROCESS_NUM)
		pool.map(run, args_list)
		pool.close()
		pool.join()
	else:
		for iindex in range(TEST_BOUND):
			if(iindex in PASS): continue
			run((iindex, reward_record, obs_record, best_record))

	if(use_multiprocess):
		args_list = list()
		for iindex in range(TEST_BOUND):
			if(iindex in PASS): continue
			args_list.append((iindex, reward_record, obs_record, best_record))
		pool = Pool(PROCESS_NUM)
		pool.map(analysis, args_list)
		pool.close()
		pool.join()
	else:
		for iindex in range(TEST_BOUND):
			if(iindex in PASS): continue
			analysis((iindex, reward_record, obs_record, best_record))
