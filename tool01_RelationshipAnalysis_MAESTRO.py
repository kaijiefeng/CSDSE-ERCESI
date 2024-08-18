import pdb
import copy
import os
import sys
from multiprocessing import Process, Lock, Manager, Pool

from config import config_global
sys.path.append("./util/")
from actor import actor_random
from sample_buffer import buffer, simple_warehouse, warehouse
from space import create_space_maestro
from evaluation_maestro import evaluation_maestro
from config_analyzer import config_self

def run(args):
	iindex, = args
	print(f"%%%%TEST{iindex} START%%%%")
	config = config_self(iindex, is_setup = True)
	nnmodel = config.nnmodel
	goal = config.goal
	target = config.target
	metrics_name = config.metrics_name
	pid = os.getpid()
	config.config_check()

	DSE_action_space = create_space_maestro(nnmodel, target = target)
	evaluation = evaluation_maestro(iindex, nnmodel, pid, DSE_action_space)

	initial_sample_warehouse = simple_warehouse(DSE_action_space, metrics_name)
	sample_warehouse = warehouse(DSE_action_space, config, metrics_name)

	count_period = 0
	upbound_for_period = 1000

	actor = actor_random()

	if(not os.path.exists("data/initial_data_warehouse_{}.csv".format(nnmodel))):
		print(f"design space scale:{DSE_action_space.get_scale()}")
		while(count_period < upbound_for_period):
			count_period = count_period + 1
			print(f"count_period:{count_period}", end="\r")

			for step in range(DSE_action_space.get_lenth()):
				DSE_action_space.sample_one_dimension(dimension_index = step, sample_index = actor.make_policy(DSE_action_space, step))
			status = DSE_action_space.get_status()
			action_list = DSE_action_space.get_action_list()
		            
			#TODO runtime return a correct value which is required to be repaired
			metrics = evaluation.evaluate(status)

			sample = list(status.values()) + list(metrics.values())
			initial_sample_warehouse.append(sample)

		initial_sample_warehouse.save("data/initial_data_warehouse_{}.csv".format(nnmodel))
		initial_sample_warehouse.save_baseline("data/baseline_{}.csv".format(nnmodel))
		initial_sample_warehouse.save_corr_spearman("data/corr_table_{}.csv".format(nnmodel))
	else:
		initial_sample_warehouse.load("data/initial_data_warehouse_{}.csv".format(nnmodel))
		initial_sample_warehouse.save_baseline("data/baseline_{}.csv".format(nnmodel))
		initial_sample_warehouse.save_corr_spearman("data/corr_table_{}.csv".format(nnmodel))

	DSE_action_space.corr_analysis("data/corr_table_{}.csv".format(nnmodel))

	sample_warehouse.load_baseline("data/baseline_{}.csv".format(nnmodel))
	for sample in initial_sample_warehouse.sample_buffer:
		sample_warehouse.update(sample)

	sample_warehouse.save("data/data_warehouse_{}.csv".format(nnmodel))
	sample_warehouse.load("data/data_warehouse_{}.csv".format(nnmodel))

	sample_buffer = sample_warehouse.create_buffer()

	#sample_buffer.print()

	print(f"%%%%TEST{iindex} END%%%%")

if __name__ == '__main__':
	os.environ["MKL_NUM_THREADS"] = "1" 
	os.environ["NUMEXPR_NUM_THREADS"] = "1" 
	os.environ["OMP_NUM_THREADS"] = "1" 

	algoname = "CORR_ANALYSIS"
	use_multiprocess = True
	global_config = config_global(is_setup = True)
	TEST_BOUND = global_config.TEST_BOUND
	PROCESS_NUM = global_config.PROCESS_NUM

	if(use_multiprocess):
		args_list = list()
		for iindex in range(TEST_BOUND):
			args_list.append((iindex,))
		pool = Pool(PROCESS_NUM)
		pool.map(run, args_list)
		pool.close()
		pool.join()
	else:
		for iindex in range(TEST_BOUND):
			run((iindex,))