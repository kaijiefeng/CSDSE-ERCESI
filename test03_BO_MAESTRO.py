from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from multiprocessing import Process, Lock, Manager, Pool
import pandas
import pdb
import numpy as np
import random
import os
import sys

from config import config_global
sys.path.append("./util/")
from space import dimension_discrete, design_space, create_space_maestro
from actor import actor_random
from evaluation_maestro import evaluation_maestro
from config_analyzer import config_self
from timer import timer
from recorder import recorder

def object_function(parameters):
	design_space = parameters["design_space"]
	config = parameters["config"]
	evaluation = parameters["evaluation"]
	record = parameters["record"]
	t = parameters["t"]

	constraints = config.constraints
	goal = config.goal
	baseline = config.baseline

	names = record.names
	action_list = list()
	for name in names:
		action_list.append(parameters[name])
	status = design_space.status_set(action_list)
	t.start("eva")
	metrics = evaluation.evaluate(status)
	t.end("eva")

	if(metrics != None):
		constraints.multi_update(metrics)
		objectvalue = metrics[goal] / baseline[goal]
		reward = objectvalue * constraints.get_punishment()
	else:
		reward = 100
	if(not record.objectvalue_list):
		record.objectvalue_list.append(objectvalue)
	else:
		pass
	best_objectvalue = record.objectvalue_list[-1]

	if(objectvalue < best_objectvalue and constraints.is_all_meet()):
		best_objectvalue = objectvalue
		print(f"objectvalue:{objectvalue}, reward:{reward}, metrics:{metrics}", end = '\r')
	record.objectvalue_list.append(best_objectvalue)
	record.multiobjecvalue_list.append([metrics["latency"], metrics["energy"]])
	
	return {"loss": reward, "status": STATUS_OK}

def run(args):
	iindex, objective_record, timecost_record, multiobjective_record = args
	print(f"%%%%%%%%%%%%%%%TEST{iindex} START%%%%%%%%%%%%%")
	seed = iindex * 10000
	np.random.seed(seed)
	random.seed(seed)
	
	config = config_self(iindex)
	constraints = config.constraints
	nnmodel = config.nnmodel
	goal = config.goal
	target = config.target
	baseline = config.baseline
	period = config.period
	config.config_check()
	pid = os.getpid()

	design_space = create_space_maestro(nnmodel, target = target)	
	evaluation = evaluation_maestro(iindex, nnmodel, pid, design_space)

	lbs = [0] * design_space.get_lenth()
	ubs = design_space.get_dimension_scale_list(has_upbound = True)
	names = list(design_space.get_status().keys())
	class record():
		def __init__(self):
			self.objectvalue_list = list()
			self.names = list(design_space.get_status().keys())
			self.multiobjecvalue_list = list()
	record = record()
	t = timer()

	space = dict()
	for lb, ub, name in zip(lbs, ubs, names):
		space[name] = hp.choice(name, range(lb, ub))
	space["design_space"] = hp.choice("design_space", [design_space])
	space["config"] = hp.choice("config", [config])
	space["evaluation"] = hp.choice("evaluation", [evaluation])
	space["record"] = hp.choice("record", [record])
	space["t"] = hp.choice("t", [t])

	trials_list = list()
	trials = Trials()
	trials_list.append(trials)
	t.start("all")
	result = fmin(fn = object_function, space = space, algo = tpe.suggest, max_evals = period, trials = trials)
	t.end("all")
	timecost_list = t.get_list("all")
	evacost = t.get_sum("eva")
	timecost_list.append(evacost)

	record.objectvalue_list.append(iindex)
	timecost_list.append(iindex)
	record.multiobjecvalue_list.append([iindex, iindex])
	objective_record.append(record.objectvalue_list)
	timecost_record.append(timecost_list)
	multiobjective_record.append(record.multiobjecvalue_list)
	print(f"%%%%TEST{iindex} END%%%%")

if __name__ == '__main__':
	algoname = "BO_MOO"
	use_multiprocess = True
	global_config = config_global()
	TEST_BOUND = global_config.TEST_BOUND
	#PROCESS_NUM = global_config.PROCESS_NUM
	PROCESS_NUM = 5
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


