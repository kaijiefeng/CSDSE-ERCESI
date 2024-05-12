import pdb
import random
import numpy as np
import pandas
import os
import sys
from multiprocessing import Process, Lock, Manager, Pool

from config import config_global
sys.path.append("./util/")
from space import dimension_discrete, design_space, create_space_maestro
from actor import actor_random
from evaluation_maestro import evaluation_maestro
from config_analyzer import config_self
from timer import timer
from recorder import recorder

def run(args):
	iindex, objective_record, timecost_record, multiobjective_record = args
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

	best_objectvalue = 1000
	best_objectvalue_list = list()
	multiobjecvalue_list = list()
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
			objectvalue = metrics[goal] / baseline[goal]
			reward = 1 / (objectvalue * constraints.get_punishment())
		else:
			reward = 0
		if(objectvalue < best_objectvalue and constraints.is_all_meet()):
			best_objectvalue = objectvalue
			#print(f"metrics:{metrics}")
		best_objectvalue_list.append(best_objectvalue)
		multiobjecvalue_list.append([metrics["latency"], metrics["energy"]])
		print(f"period:{count_period}, best:{best_objectvalue}, objectvalue:{objectvalue}, reward:{reward}", end = '\r')

	print(f"%%%%TEST{iindex} END%%%%")
	t.end("all")
	timecost_list = t.get_list("all")
	evacost = t.get_sum("eva")
	timecost_list.append(evacost)

	best_objectvalue_list.append(iindex)
	timecost_list.append(iindex)
	multiobjecvalue_list.append([iindex, iindex])
	objective_record.append(best_objectvalue_list)
	timecost_record.append(timecost_list)
	multiobjective_record.append(multiobjecvalue_list)

if __name__ == '__main__':
	algoname = "RGS_MOO"
	use_multiprocess = True
	#use_multiprocess = False
	global_config = config_global()
	TEST_BOUND = global_config.TEST_BOUND
	PROCESS_NUM = global_config.PROCESS_NUM
	SCEN_TYPE = global_config.SCEN_TYPE
	SCEN_NUM = global_config.SCEN_NUM
	PASS = global_config.PASS

	args_list = list()
	objective_record = Manager().list()
	timecost_record = Manager().list()
	multiobjective_record = Manager().list()

	if(use_multiprocess):
		args_list = list()
		for iindex in range(TEST_BOUND):
			if(iindex in PASS): continue
			args_list.append((iindex, objective_record, timecost_record, multiobjective_record))
		pool = Pool(PROCESS_NUM)
		pool.map(run, args_list)
		pool.close()
		pool.join()
	else:
		for iindex in range(TEST_BOUND):
			if(iindex in PASS): continue
			run((iindex, objective_record, timecost_record))

	recorder(algoname, global_config, objective_record, timecost_record, multiobjective_record)

