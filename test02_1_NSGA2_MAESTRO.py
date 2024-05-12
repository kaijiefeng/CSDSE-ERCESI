import numpy as np
import geatpy as ea
import random
import pandas
from multiprocessing import Process, Lock, Manager, Pool
import pdb
import time
import os
import sys

from config import config_global
sys.path.append("./util/")
from space import create_space_maestro
from evaluation_maestro import evaluation_maestro
from config_analyzer import config_self
from timer import timer
from recorder import recorder

NIND = 50

class DSE(ea.Problem):
	def __init__(self, config, design_space, evaluator, t):
		self.config = config
		self.goal = self.config.goal
		self.constraints = self.config.constraints
		self.baseline = self.config.baseline
		self.design_space = design_space
		self.evaluator = evaluator
		self.t = t

		self.best_objectvalue = 1000
		self.best_objectvalue_list = list()
		self.multiobjecvalue_list = list()

		self.Nindex = 0

		name = "DSE"
		if(self.config.goal == "latency" or self.config.goal == "energy"):
			M = 1
		if(self.config.goal == "edp"):
			M = 2 # the dimension of object function
		maxormins = [1] * M # 1 means to minmize the ob function

		Dim = self.design_space.get_lenth()
		varTypes = [1] * Dim
		lb = [0] * Dim
		ub = self.design_space.get_dimension_scale_list(has_upbound = False)

		lbin = [1] * Dim
		ubin = [1] * Dim

		ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

	def aimFunc(self, pop):
		vars = pop.Phen # Phen is the variables matrix
		lenth = len(vars)
		print(f"vvvvvlenth:{lenth}")
		vec_latency = np.zeros(lenth).reshape(-1,1)
		vec_energy = np.zeros(lenth).reshape(-1,1)
		vec_area = np.zeros(lenth).reshape(-1,1)
		vec_power = np.zeros(lenth).reshape(-1,1)
		vec_cnt_pes = np.zeros(lenth).reshape(-1,1)
		vec_l1_mem = np.zeros(lenth).reshape(-1,1)
		vec_l2_mem = np.zeros(lenth).reshape(-1,1)
		vec_edp = np.zeros(lenth).reshape(-1,1)

		for index in range(lenth):
			self.Nindex += 1
			action_list = list(vars[index].astype(int))
			status = self.design_space.status_set(action_list)
			self.t.start("eva")
			metrics = self.evaluator.evaluate(status)
			self.t.end("eva")

			if(metrics != None):
				self.constraints.multi_update(metrics)
				objectvalue = metrics[self.goal] / self.baseline[self.goal]
				reward = 1 / (objectvalue * self.constraints.get_punishment())
			else:
				reward = 0
			if(objectvalue < self.best_objectvalue and self.constraints.is_all_meet()):
				self.best_objectvalue = objectvalue
			self.best_objectvalue_list.append(self.best_objectvalue)
			self.multiobjecvalue_list.append([metrics["latency"], metrics["energy"]])
			print(f"Nindex:{self.Nindex}, objectvalue:{objectvalue}, reward:{reward}, bset:{self.best_objectvalue}", end = '\r')

			vec_latency[index] = metrics["latency"]
			vec_energy[index] = metrics["energy"]
			vec_area[index] = metrics["area"] - self.config.AREA_THRESHOLD
			vec_power[index] = metrics["power"] - self.config.POWER_THRESHOLD
			vec_cnt_pes[index] = metrics["cnt_pes"] - self.config.NUMPES_THRESHOLD
			vec_l1_mem[index] = metrics["l1_mem"] - self.config.L1SIZE_THRESHOLD
			vec_l2_mem[index] = metrics["l2_mem"] - self.config.L2SIZE_THRESHOLD
			vec_edp[index] = metrics["edp"]

		if(self.config.goal == "latency"):
			pop.ObjV = vec_latency
		elif(self.config.goal == "energy"):
			pop.ObjV = vec_energy
		elif(self.config.goal == "edp"):
			pop.ObjV = np.hstack([vec_latency, vec_energy])

		pop.CV = np.hstack([vec_cnt_pes,
							vec_l1_mem,
							vec_l2_mem,
							vec_area,
							vec_power])

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
	config.config_check()
	pid = os.getpid()

	design_space = create_space_maestro(nnmodel, target = target)	
	evaluator = evaluation_maestro(iindex, nnmodel, pid, design_space)
	t = timer()

	MAXGEN = int(config.period / NIND)
	print(f"maxgen:{MAXGEN}")

	problem = DSE(config, design_space, evaluator, t)
	Encoding = "RI" 
	Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)
	population = ea.Population(Encoding, Field, NIND)
	#myalgorithm = ea.soea_EGA_templet(problem, population)
	#myalgorithm = ea.soea_SEGA_templet(problem, population)
	#myalgorithm = ea.soea_DE_rand_1_bin_templet(problem, population)
	#myalgorithm = ea.soea_DE_best_1_bin_templet(problem, population)
	#myalgorithm = ea.soea_SGA_templet(problem, population)
	#myalgorithm = ea.moea_NSGA2_DE_templet(problem, population)
	myalgorithm = ea.moea_NSGA2_templet(problem, population)

	myalgorithm.MAXGEN = MAXGEN
	myalgorithm.MAXEVALS = config.period
	#myalgorithm.mutOper.F = 0.5
	#myalgorithm.recOper.XOVR = 0.7
	myalgorithm.drawing = 0

	t.start("all")
	[BestIndi, population] = myalgorithm.run()
	t.end("all")
	timecost_list = t.get_list("all")
	evacost = t.get_sum("eva")
	timecost_list.append(evacost)

	problem.best_objectvalue_list.append(iindex)
	timecost_list.append(iindex)
	problem.multiobjecvalue_list.append([iindex, iindex])
	objective_record.append(problem.best_objectvalue_list)
	timecost_record.append(timecost_list)
	multiobjective_record.append(problem.multiobjecvalue_list)

	print(f"**************TEST{iindex} END***********")

if __name__ == '__main__':
	algoname = "NSGA2_MOO"
	use_multiprocess = True
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
			#if(iindex != 1): continue
			run((iindex, objective_record, timecost_record, multiobjective_record))

	recorder(algoname, global_config, objective_record, timecost_record, multiobjective_record)
