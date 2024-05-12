import pandas
import pdb
import numpy as np
from config import config_maestro
from space import create_space_maestro

global_config = config_maestro()
design_space = create_space_maestro(global_config.nnmodel, global_config.target)
TEST_BOUND = global_config.TEST_BOUND
PROCESS_NUM = global_config.PROCESS_NUM
SCEN_TYPE = global_config.SCEN_TYPE
SCEN_NUM = global_config.SCEN_NUM
PASS = global_config.PASS

obs_distance_list = list()
metric_distance_list = list()

#dtype = "Euclidean"
dtype = "Hamming"

for iindex in range(TEST_BOUND):
	if(iindex < PASS): continue
	print(f"processing iindex:{iindex}")
	config = config_maestro(iindex)
	buffer_obs_path = "record/buffer/obs/CSDSE_obs_{}_{}_{}_i={}.csv".format(config.goal, config.nnmodel, config.target, iindex)
	buffer_metrics_path = "record/buffer/metrics/CSDSE_metric_{}_{}_{}_i={}.csv".format(config.goal, config.nnmodel, config.target, iindex)

	obs_samples_list = pandas.read_csv(buffer_obs_path).values.tolist()
	metrics_samples_list = pandas.read_csv(buffer_metrics_path).values.tolist()
	#pdb.set_trace()

	list_lenth = len(obs_samples_list)#len(metrics_samples_list)
	para_lenth = len(obs_samples_list[0])
	metric_lenth = len(metrics_samples_list[0])
	distance_cnt = 0
	obs_distance_sum = 0
	metric_distance_sum = 0
	for i in range(0, list_lenth):
		for j in range(i+1, list_lenth):
			obs_distance = 0
			obs_sample_a = obs_samples_list[i]
			obs_sample_b = obs_samples_list[j]
			for para_a, para_b in zip(obs_sample_a, obs_sample_b):
				obs_distance += (para_a - para_b)**2
			obs_distance = obs_distance**0.5
			obs_distance_sum += obs_distance

			metric_distance = 0
			metric_sample_a = metrics_samples_list[i]
			metric_sample_b = metrics_samples_list[j]
			for metric_a, metric_b in zip(metric_sample_a, metric_sample_b):
				metric_distance += (metric_a - metric_b)**2
			metric_distance = metric_distance**0.5
			metric_distance_sum += metric_distance

			distance_cnt += 1
			#pdb.set_trace()
	obs_distance_avg = obs_distance_sum/distance_cnt
	metric_distance_avg = metric_distance_sum/distance_cnt
	#obs_distance_avg = obs_distance_avg/para_lenth
	#metric_distance_avg = metric_distance_avg/metric_lenth
	obs_distance_list.append(obs_distance_avg)
	metric_distance_list.append(metric_distance_avg)
	print(f"distance computing: n:{list_lenth}, cnt:{distance_cnt}, avg_dis:{obs_distance_avg}, avg_metric:{metric_distance_avg}")

distance_path = "record/buffer/CSDSE_distance_{}_{}_{}_i={}.csv".format(config.goal, config.nnmodel, config.target, iindex)

record = list()
record.append(obs_distance_list)
record.append(metric_distance_list)
record = np.array(record)
record = pandas.DataFrame(record)
record.to_csv(distance_path, index = None, header = None)