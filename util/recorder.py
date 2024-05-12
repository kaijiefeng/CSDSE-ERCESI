import numpy as np
import pandas
import os

def	recorder(algoname, global_config, objective_record, timecost_record, multiobjective_record):
	SCEN_NUM = global_config.SCEN_NUM
	SCEN_TYPE = global_config.SCEN_TYPE
	PASS = global_config.PASS
	objective_path = "./record/objectvalue/{}_{}.csv".format(algoname, global_config.goal)
	py_objective_record = list()
	py_objective_record = objective_record[0:len(objective_record)]
	py_objective_record.sort(key = lambda olist:olist[-1])
	py_objective_record = np.array(py_objective_record).T
	objective_df = pandas.DataFrame(py_objective_record)
	actual_SCEN_TYPE = SCEN_TYPE - int(len(PASS)/SCEN_NUM)
	for scen in range(actual_SCEN_TYPE):
		objective_df["avg_{}".format(scen)] = objective_df.iloc[:, scen*SCEN_NUM:(scen+1)*SCEN_NUM].mean(axis=1)
		#objective_df["min_{}".format(scen)] = objective_df.iloc[:, scen*SCEN_NUM:(scen+1)*SCEN_NUM].min(axis=1)
	objective_df.to_csv(objective_path, index = None)

	timecost_path = "./record/timecost/{}_{}.csv".format(algoname, global_config.goal)
	py_timecost_record = list()
	py_timecost_record = timecost_record[0:len(timecost_record)]
	py_timecost_record.sort(key = lambda olist:olist[-1])
	py_timecost_record = np.array(py_timecost_record).T
	timecost_df = pandas.DataFrame(py_timecost_record)
	for scen in range(actual_SCEN_TYPE):
		timecost_df["avg_{}".format(scen)] = timecost_df.iloc[:, scen*SCEN_NUM:(scen+1)*SCEN_NUM].mean(axis=1)
	timecost_df.to_csv(timecost_path, index = None)

	multiobjective_path = "./record/multiobjectvalue/{}_{}.csv".format(algoname, global_config.goal)
	py_multiobjective_record = list()
	py_multiobjective_record = multiobjective_record[0:len(multiobjective_record)]
	py_multiobjective_record.sort(key = lambda olist:olist[-1][0])
	is_dominate = lambda a,b:(a[0]<b[0] and a[1]<b[1])
	for scen in range(actual_SCEN_TYPE):
		origin_multiobjective_list = list()
		pareto_list = list()
		for item in range(scen*SCEN_NUM, (scen+1)*SCEN_NUM):
			py_multiobjective_record[item].pop(-1)
			origin_multiobjective_list.extend(py_multiobjective_record[item])
		while(origin_multiobjective_list):
			this_point = origin_multiobjective_list.pop(0)
			if(this_point == [-114, -514]): continue
			is_onpareto = True
			for index in range(len(origin_multiobjective_list)):
				point = origin_multiobjective_list[index]
				if(point == [-114, -514]): continue
				if(is_dominate(point, this_point)): is_onpareto = False
				elif(is_dominate(this_point, point)): origin_multiobjective_list[index] = [-114, -514]
				else: pass
			if(is_onpareto): pareto_list.append(this_point)
		pareto_list.append([scen, scen])
		pareto_list = np.array(pareto_list).T
		pareto_df = pandas.DataFrame(pareto_list)	
		pareto_df.to_csv(multiobjective_path, mode="a", header=False, index=False)
