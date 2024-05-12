from subprocess import Popen, PIPE
import pandas
import numpy as np
import os, sys
import pdb
import copy
script_dir = os.path.dirname(__file__)
module_path = os.path.abspath(os.path.join(script_dir, '../'))
if module_path not in sys.path:
	sys.path.insert(0, module_path)

class evaluation_maestro():
	def __init__(self, iindex, nnmodel, pid, space, is_adaptive = True, is_const = False):
		self.iindex = iindex
		self.nnmodel = nnmodel
		self.pid = pid
		self.is_adaptive = is_adaptive
		self.is_const = is_const

		self.layer_list = space.layer_name
		self.type_list = space.type_list
		self.stride_list = space.stride_list
		self.block_list = space.block_list

	def evaluate(self, status, save_files = False):
		if(self.nnmodel == 'VGG16'): model_filename = './desc/model/vgg16_model.m'
		elif(self.nnmodel == 'MobileNetV2'): model_filename = './desc/model/MobileNetV2_model.m'
		elif(self.nnmodel == 'MnasNet'): model_filename = './desc/model/mnasnet_model.m'
		elif(self.nnmodel == 'ResNet50'): model_filename = './desc/model/Resnet50_model.m'
		#elif(self.nnmodel == 'Transformer'): model_filename = './desc/model/Transformer_Complete_model.m'
		elif(self.nnmodel == 'Transformer'): model_filename = './desc/model/Transformer_Complete_model_littleRS.m'
		elif(self.nnmodel == 'GNMT'): model_filename = './desc/model/gnmt_model.m'
		else: pass
		## get the layer name and type from model file
		layer_list = []
		type_list = []
		layer_list = copy.deepcopy(self.layer_list)
		type_list = copy.deepcopy(self.type_list)
		stride_list = copy.deepcopy(self.stride_list)
		block_list = copy.deepcopy(self.block_list)

		if(self.is_adaptive and not self.is_const):
			#### fill dataflow.m  ####
			## define the parallelism ##
			dim_num = status['dim_num']
			dim_out = status['dim_out']
			dim_mid = status['dim_mid']
			dim_in = status['dim_in']
			pname_list = [('C', 'p_c'), ('K', 'p_k'), ('X\'', 'p_x'), ('Y\'', 'p_y'), ('R', 'p_r'), ('S', 'p_s')]
			p_list = []
			for pname in pname_list:
				p_list.append({'name':pname[0], 'value':status[pname[1]]})
			p_list.sort(reverse = True, key = lambda p:p['value'])
			para_list = []
			if(dim_num==3):
				para_list.append((p_list[0]['name'], dim_out))
				para_list.append((p_list[1]['name'], dim_mid))
				para_list.append((p_list[2]['name'], dim_in))	
			elif(dim_num==2):
				para_list.append((p_list[0]['name'], dim_mid))
				para_list.append((p_list[1]['name'], dim_in))
			elif(dim_num==1):
				para_list.append((p_list[0]['name'], dim_in))
			paraname_list = []
			for para in para_list:
				paraname_list.append(para[0])

			## define the tiling and order ##
			## outer tiling parse ##
			t_status_layer = []
			for layer, stride, block_index in zip(layer_list, stride_list, block_list):
				t_list_c, t_list_k, t_list_x, t_list_y = [], [], [], []
				t_status = {}
				tname_list_c = ['t_c_d1', 't_c_d2', 't_c_d3']
				for tname_c in tname_list_c: t_list_c.append(status['{}_{}'.format(tname_c, block_index)])
				t_list_c.sort(reverse = True)
				tname_list_c = ['t_c_out', 't_c_mid', 't_c_in']
				for tname_c, t_c in zip(tname_list_c, t_list_c): t_status['{}_{}'.format(tname_c, layer)] = t_c####
				tname_list_k = ['t_k_d1', 't_k_d2', 't_k_d3']
				for tname_k in tname_list_k: t_list_k.append(status['{}_{}'.format(tname_k, block_index)])
				t_list_k.sort(reverse = True)
				tname_list_k = ['t_k_out', 't_k_mid', 't_k_in']
				for tname_k, t_k in zip(tname_list_k, t_list_k): t_status['{}_{}'.format(tname_k, layer)] = t_k####
				tname_list_x = ['t_x_d1', 't_x_d2', 't_x_d3']
				for tname_x in tname_list_x: t_list_x.append(status['{}_{}'.format(tname_x, block_index)])
				t_list_x.sort(reverse = True)
				tname_list_x = ['t_x_out', 't_x_mid', 't_x_in']
				#for tname_x, t_x in zip(tname_list_x, t_list_x): t_status['{}_{}'.format(tname_x, layer)] = t_x####
				for tname_x, t_x in zip(tname_list_x, t_list_x): 
					if(stride < 2): t_status['{}_{}'.format(tname_x, layer)] = t_x####
					else: 
						if(tname_x == 't_x_out'): t_status['{}_{}'.format(tname_x, layer)] = t_x
						elif(tname_x == 't_x_mid'): t_status['{}_{}'.format(tname_x, layer)] = max(1, t_x - stride)
						elif(tname_x == 't_x_in'): t_status['{}_{}'.format(tname_x, layer)] = max(1, t_x - stride - stride)

				tname_list_y = ['t_y_d1', 't_y_d2', 't_y_d3']
				for tname_y in tname_list_y: t_list_y.append(status['{}_{}'.format(tname_y, block_index)])
				t_list_y.sort(reverse = True)
				tname_list_y = ['t_y_out', 't_y_mid', 't_y_in']
				#for tname_y, t_y in zip(tname_list_y, t_list_y): t_status['{}_{}'.format(tname_y, layer)] = t_y####
				for tname_y, t_y in zip(tname_list_y, t_list_y):
					if(self.nnmodel != "GNMT"): 
						if(stride <2): t_status['{}_{}'.format(tname_y, layer)] = t_y####
						else:
							if(tname_y == 't_y_out'): t_status['{}_{}'.format(tname_y, layer)] = t_y
							elif(tname_y == 't_y_mid'): t_status['{}_{}'.format(tname_y, layer)] = max(1, t_y - stride)
							elif(tname_y == 't_y_in'): t_status['{}_{}'.format(tname_y, layer)] = max(1, t_y - stride - stride)
					#### MAESTRO BUG: the shape of GNMT it too large that may incur overflow of MAESTRO if cluter=1 and all tiling size=1
					#### Therefore, we manually scale up the tiling size of Y up at least 2
					else:
						t_status['{}_{}'.format(tname_y, layer)] = max(2,t_y)

				t_status_layer.append(t_status)

			## outer loop parse ##
			ot_list_out_layer, ot_list_mid_layer, ot_list_in_layer = [], [], []
			for layer, t_status, block_index in zip(layer_list, t_status_layer, block_list):
				ot_list_out, ot_list_mid, ot_list_in = [], [], []
				oname_list_out = [('C', 'o_c_out', 't_c_out') , ('K', 'o_k_out', 't_k_out'), ('X\'', 'o_x_out', 't_x_out'), ('Y\'', 'o_y_out', 't_y_out'), \
				('R', 'o_r_out', 't_r_out'), ('S', 'o_s_out', 't_r_out')]
				for oname in oname_list_out:
					if(oname[0] in ['C', 'K', 'X\'', 'Y\'']):
						ot_list_out.append({'name':oname[0], 'order':status['{}_{}'.format(oname[1],block_index)], 'tiling':t_status['{}_{}'.format(oname[2],layer)], 'offset':t_status['{}_{}'.format(oname[2],layer)]})
					elif(oname[0] == 'R'):
						ot_list_out.append({'name':oname[0], 'order':status['{}_{}'.format(oname[1],block_index)], 'tiling':'Sz(R)', 'offset':'Sz(R)'})
					elif(oname[0] == 'S'):
						ot_list_out.append({'name':oname[0], 'order':status['{}_{}'.format(oname[1],block_index)], 'tiling':'Sz(S)', 'offset':'Sz(S)'})
					else:
						print(f"tiling/ordering only support C,K,X,Y,R,S")
				ot_list_out.sort(reverse = True, key = lambda ot:ot['order'])
				## middle loop parse ##
				oname_list_mid = [('C', 'o_c_mid', 't_c_mid') , ('K', 'o_k_mid', 't_k_mid'), ('X\'', 'o_x_mid', 't_x_mid'), ('Y\'', 'o_y_mid', 't_y_mid'), \
				('R', 'o_r_mid', 't_r_mid'), ('S', 'o_s_mid', 't_s_mid')]
				for oname in oname_list_mid:
					if(oname[0] in ['C', 'K', 'X\'', 'Y\'']):
						ot_list_mid.append({'name':oname[0], 'order':status['{}_{}'.format(oname[1],block_index)], 'tiling':t_status['{}_{}'.format(oname[2],layer)], 'offset':t_status['{}_{}'.format(oname[2],layer)]})
					elif(oname[0] == 'R'):
						ot_list_mid.append({'name':oname[0], 'order':status['{}_{}'.format(oname[1],block_index)], 'tiling':'Sz(R)', 'offset':'Sz(R)'})
					elif(oname[0] == 'S'):
						ot_list_mid.append({'name':oname[0], 'order':status['{}_{}'.format(oname[1],block_index)], 'tiling':'Sz(S)', 'offset':'Sz(S)'})
					else:
						print(f"tiling/ordering only support C,K,X,Y,R,S")
				ot_list_mid.sort(reverse = True, key = lambda ot:ot['order'])
				## inner loop parse ##
				oname_list_in = [('C', 'o_c_in', 't_c_in') , ('K', 'o_k_in', 't_k_in'), ('X\'', 'o_x_in', 't_x_in'), ('Y\'', 'o_y_in', 't_y_in'), \
				('R', 'o_r_in', 't_r_in'), ('S', 'o_s_in', 't_s_in')]
				for oname in oname_list_in:
					if(oname[0] in ['C', 'K', 'X\'', 'Y\'']):
						ot_list_in.append({'name':oname[0], 'order':status['{}_{}'.format(oname[1],block_index)], 'tiling':t_status['{}_{}'.format(oname[2],layer)], 'offset':t_status['{}_{}'.format(oname[2],layer)]})
					elif(oname[0] == 'R'):
						ot_list_in.append({'name':oname[0], 'order':status['{}_{}'.format(oname[1],block_index)], 'tiling':'Sz(R)', 'offset':'Sz(R)'})
					elif(oname[0] == 'S'):
						ot_list_in.append({'name':oname[0], 'order':status['{}_{}'.format(oname[1],block_index)], 'tiling':'Sz(S)', 'offset':'Sz(S)'})
					else:
						print(f"tiling/ordering only support C,K,X,Y,R,S")
				ot_list_in.sort(reverse = True, key = lambda ot:ot['order'])
				ot_list_out_layer.append(ot_list_out)
				ot_list_mid_layer.append(ot_list_mid)
				ot_list_in_layer.append(ot_list_in)

			## modify the dataflow file ##
			with open(model_filename, 'r') as mdfile:
				with open('./desc/dataflow_{}_{}.m'.format(self.iindex, self.pid), 'w') as dffile:
					lines = mdfile.readlines()
					for line in lines:
						dffile.write(line)
						if(line.find('Dimensions') != -1):
							layer = layer_list.pop(0)
							ltype = type_list.pop(0)
							t_status = t_status_layer.pop(0)
							ot_list_out = ot_list_out_layer.pop(0)
							ot_list_mid = ot_list_mid_layer.pop(0)
							ot_list_in = ot_list_in_layer.pop(0)
							dffile.write('Dataflow {\n')
							if(ltype == 'CONV'):
								for index, para in enumerate(para_list):
									if(index != 0): dffile.write('Cluster({}, P);\n'.format(para[1]))
									if(index == 0): 
										if(dim_num == 3): ot_list = ot_list_out
										elif(dim_num == 2): ot_list = ot_list_mid
										elif(dim_num == 1): ot_list = ot_list_in 
									elif(index == 1): 
										if(dim_num == 3): ot_list = ot_list_mid
										elif(dim_num == 2): ot_list = ot_list_in
									elif(index == 2): 
										if(dim_num == 3): ot_list = ot_list_in
									else: pass
									for ot in ot_list:
										if(ot['name'] == para[0]): dffile.write('SpatialMap({}, {}) {};\n'.format(ot['tiling'], ot['offset'], ot['name']))
										elif(ot['name'] in ['C', 'K', 'X\'', 'Y\'']): dffile.write('TemporalMap({}, {}) {};\n'.format(ot['tiling'], ot['offset'], ot['name']))
										elif(ot['name'] == 'R'): dffile.write('TemporalMap(Sz(R), Sz(R)) R;\n')
										elif(ot['name'] == 'S'): dffile.write('TemporalMap(Sz(S), Sz(S)) S;\n')
										else: pass
							elif(ltype == 'DSCONV'):
								para_list_copy = copy.deepcopy(para_list)
								for index, para in enumerate(para_list):
									if(para[0] == 'K'):
										if(dim_num == 1): 
											para_list_copy.pop(index)
											para_list_copy.append(('C', dim_out))
										else:
											para_list_copy.pop(index)

								for index, para in enumerate(para_list_copy):
									if(index != 0): dffile.write('Cluster({}, P);\n'.format(para[1]))
									if(index == 0): 
										if(dim_num == 3): ot_list = ot_list_out
										elif(dim_num == 2): ot_list = ot_list_mid
										elif(dim_num == 1): ot_list = ot_list_in 
									elif(index == 1): 
										if(dim_num == 3): ot_list = ot_list_mid
										elif(dim_num == 2): ot_list = ot_list_in
									elif(index == 2): 
										if(dim_num == 3): ot_list = ot_list_in
									else: pass
									for ot in ot_list:
										if(ot['name'] == para[0]): dffile.write('SpatialMap({}, {}) {};\n'.format(ot['tiling'], ot['offset'], ot['name']))
										elif(ot['name'] in ['C', 'X\'', 'Y\'']): dffile.write('TemporalMap({}, {}) {};\n'.format(ot['tiling'], ot['offset'], ot['name']))
										elif(ot['name'] == 'R'): dffile.write('TemporalMap(Sz(R), Sz(R)) R;\n')
										elif(ot['name'] == 'S'): dffile.write('TemporalMap(Sz(S), Sz(S)) S;\n')
										elif(ot['name'] == 'K'): pass
										else: pass		
							dffile.write('}\n')
		else:
			pass

		#####  fill hw.m  #####
		if(not self.is_const):
			num_pes = 1
			for para in para_list:
				num_pes *= para[1]
		else:
			num_pes = status['num_pes']
		l1_size = status['l1_size']
		l2_size = status['l2_size']
		noc_bw = status['noc_bw']
		offchip_bw = status['offchip_bw']

		with open('./desc/hw_{}_{}.m'.format(self.iindex, self.pid), 'w') as hwfile:
			hwfile.write('num_pes: {}\n'.format(num_pes))
			hwfile.write('l1_size_cstr: {}\n'.format(l1_size))
			hwfile.write('l2_size_cstr: {}\n'.format(l2_size))
			hwfile.write('noc_bw_cstr: {}\n'.format(noc_bw))
			hwfile.write('offchip_bw_cstr: {}\n'.format(offchip_bw))

		#### lanuch the evaluation model  ####
		command = ["../maestro/maestro",
				"--HW_file=./desc/hw_{}_{}.m".format(self.iindex, self.pid),
				"--Mapping_file=./desc/dataflow_{}_{}.m".format(self.iindex, self.pid),
				"--print_res=false", "--print_res_csv_file=true", "--print_log_file=false"]

		process = Popen(command, stdout=PIPE, stderr=PIPE)
		stdout, stderr = process.communicate()
		process.wait()

		####  get metrics from output file  ####
		try:
			df = pandas.read_csv("./dataflow_{}_{}.csv".format(self.iindex, self.pid))
			layer_name = df[" Layer Number"]
			runtime_perlayer = np.array(df[" Runtime (Cycles)"]).reshape(-1, 1)
			throughput_perlayer = np.array(df[" Throughput (MACs/Cycle)"]).reshape(-1, 1)
			energy_perlayer = np.array(df[" Activity count-based Energy (nJ)"]).reshape(-1, 1)
			area_perlayer = np.array(df[" Area"]).reshape(-1, 1)
			power_perlayer = np.array(df[" Power"]).reshape(-1, 1)
			l1_size_perlayer = np.array(df[" L1 SRAM Size Req (Bytes)"]).reshape(-1, 1)
			l2_size_perlayer = np.array(df["  L2 SRAM Size Req (Bytes)"]).reshape(-1, 1)
			mac_perlayer = np.array(df[" Num MACs"]).reshape(-1, 1)

			runtime = np.sum(runtime_perlayer)
			energy = np.sum(energy_perlayer)
			mac = np.sum(mac_perlayer)
			area = np.max(area_perlayer)
			l1_size = np.max(l1_size_perlayer)
			l2_size = np.max(l2_size_perlayer)
			power = np.mean(power_perlayer)
			throughput = mac/runtime
			edp = runtime*energy

			if(not save_files):
				os.remove("./dataflow_{}_{}.csv".format(self.iindex, self.pid))  if os.path.exists("./dataflow_{}_{}.csv".format(self.iindex, self.pid)) else None
				os.remove("./desc/dataflow_{}_{}.m".format(self.iindex, self.pid))  if os.path.exists("./desc/dataflow_{}_{}.m".format(self.iindex, self.pid)) else None
				os.remove("./desc/hw_{}_{}.m".format(self.iindex, self.pid))  if os.path.exists("./desc/hw_{}_{}.m".format(self.iindex, self.pid)) else None

			metrics = {
				'latency':runtime, # unit: cycle
				#'throughput':throughput, # unit: MACs/cycle
				'energy':energy, # unit: nJ
				'area':area, # unit: um^2
				'power':power/1000, # unit: W (1000mW)
				'cnt_pes':num_pes, # unit: /
				'l1_mem':l1_size, # unit: Byte
				'l2_mem':l2_size, # unit: Byte
				#'mac':mac,
				'edp':edp
			}

			#print(f"metrics:{metrics}")

			return metrics
		except:
			print(f"current status can't be evaluated")
			return None


if __name__ == '__main__':
	model = 'VGG16'
	is_adaptive = True  ## do all layers share the same dataflow
	is_const = False ## is_const = True, we use categorical vairable to describe the dataflow, that is, xp, xyp, yrp, cp, ckp
	                ## is_const = False, we use vector to describe the dataflow, that is, {dim_num, dim_out, dim_mid, ..., p_c, p_k, ..., t_c, t_k,...}
	maestro = evaluation_maestro(iindex =0, nnmodel = model, pid = 0, is_adaptive = True, is_const = False)
	status_example = {}

	if(not is_adaptive and is_const):
		if(model == 'VGG16'):
			status_example = {
				'num_pes':1024,
				'l1_size':100,
				'l2_size':3000,
				'noc_bw':1000,
				'offchip_bw':50,
				'dataflow':0 ## cp=0, xp=1, yxp=2, yrp=3, kcp=4 
			}	
	elif(is_adaptive and is_const):
		if(model == 'VGG16'):
			status_example = {
				'num_pes':1024,
				'l1_size':100,
				'l2_size':3000,
				'noc_bw':1000,
				'offchip_bw':50,
				'dataflow_CONV1':0,## cp=0, xp=1, yxp=2, yrp=3, kcp=4
				'dataflow_CONV2':0,
				'dataflow_CONV3':0,
				'dataflow_CONV4':0,
				'dataflow_CONV5':0,
				'dataflow_CONV6':0,
				'dataflow_CONV7':0,
				'dataflow_CONV8':0,
				'dataflow_CONV9':0,
				'dataflow_CONV10':0,
				'dataflow_CONV11':0,
				'dataflow_CONV12':0,
				'dataflow_CONV13':0
			}
	elif(not is_adaptive and not is_const):
		if(model == 'VGG16'):
			status_example = {
				#'num_pes':1024,
				'l1_size':100,
				'l2_size':3000,
				'noc_bw':1000,
				'offchip_bw':50,
				'dim_num':2,
				'dim_out':30,
				'dim_mid':30,
				'dim_in':10,
				'p_c':6,
				'p_k':5,
				'p_x':4,
				'p_y':3,
				'p_r':2,
				'p_s':1,

				'o_c_out':6,
				'o_k_out':5,
				'o_x_out':4,
				'o_y_out':3,
				'o_r_out':2,
				'o_s_out':1,
				'o_c_mid':6,
				'o_k_mid':5,
				'o_x_mid':4,
				'o_y_mid':3,
				'o_r_mid':2,
				'o_s_mid':1,
				'o_c_in':6,
				'o_k_in':5,
				'o_x_in':4,
				'o_y_in':3,
				'o_r_in':2,
				'o_s_in':1,

				't_c_d1':4,
				't_k_d1':4,
				't_x_d1':4,
				't_y_d1':4,
				't_c_d2':6,
				't_k_d2':6,
				't_x_d2':6,
				't_y_d2':6,
				't_c_d3':1,
				't_k_d3':1,
				't_x_d3':1,
				't_y_d3':1,
			}
	elif(is_adaptive and not is_const):
		if(model == 'VGG16'):
			status_example = {
				'l1_size': 64000, 'l2_size': 128000, 
				'noc_bw': 256, 'offchip_bw': 64, 
				'dim_num': 3, 'dim_out': 28, 'dim_mid': 48, 'dim_in': 48, 
				'p_c': 4, 'p_k': 2, 'p_x': 6, 'p_y': 1, 'p_r': 2, 'p_s': 2, 

				'o_c_out_CONV1': 5, 'o_k_out_CONV1': 3, 'o_x_out_CONV1': 5, 'o_y_out_CONV1': 1, 'o_r_out_CONV1': 5, 'o_s_out_CONV1': 3,
				 'o_c_mid_CONV1': 1, 'o_k_mid_CONV1': 5, 'o_x_mid_CONV1': 3, 'o_y_mid_CONV1': 3, 'o_r_mid_CONV1': 6, 'o_s_mid_CONV1': 5, 
				'o_c_in_CONV1': 4, 'o_k_in_CONV1': 6, 'o_x_in_CONV1': 5, 'o_y_in_CONV1': 2, 'o_r_in_CONV1': 1, 'o_s_in_CONV1': 1, 
				't_c_d1_CONV1': 1, 't_c_d2_CONV1': 3, 't_c_d3_CONV1': 1,
				 't_k_d1_CONV1': 16, 't_k_d2_CONV1': 32, 't_k_d3_CONV1': 1, 
				't_x_d1_CONV1': 1, 't_x_d2_CONV1': 3, 't_x_d3_CONV1': 1, 
				't_y_d1_CONV1': 74, 't_y_d2_CONV1': 2, 't_y_d3_CONV1': 1, 

				'o_c_out_CONV2': 1, 'o_k_out_CONV2': 2, 'o_x_out_CONV2': 4, 'o_y_out_CONV2': 3, 'o_r_out_CONV2': 6, 'o_s_out_CONV2': 5, 
				'o_c_mid_CONV2': 2, 'o_k_mid_CONV2': 5, 'o_x_mid_CONV2': 2, 'o_y_mid_CONV2': 5, 'o_r_mid_CONV2': 2, 'o_s_mid_CONV2': 3, 
				'o_c_in_CONV2': 3, 'o_k_in_CONV2': 6, 'o_x_in_CONV2': 1, 'o_y_in_CONV2': 6, 'o_r_in_CONV2': 2, 'o_s_in_CONV2': 3, 
				't_c_d1_CONV2': 32, 't_c_d2_CONV2': 1, 't_c_d3_CONV2': 1, 
				't_k_d1_CONV2': 32, 't_k_d2_CONV2': 32, 't_k_d3_CONV2': 1, 
				't_x_d1_CONV2': 2, 't_x_d2_CONV2': 6, 't_x_d3_CONV2': 1, 
				't_y_d1_CONV2': 6, 't_y_d2_CONV2': 74, 't_y_d3_CONV2': 1, 

				'o_c_out_CONV3': 6, 'o_k_out_CONV3': 6, 'o_x_out_CONV3': 1, 'o_y_out_CONV3': 4, 'o_r_out_CONV3': 6, 'o_s_out_CONV3': 6, 
				'o_c_mid_CONV3': 5, 'o_k_mid_CONV3': 6, 'o_x_mid_CONV3': 4, 'o_y_mid_CONV3': 3, 'o_r_mid_CONV3': 6, 'o_s_mid_CONV3': 4, 
				'o_c_in_CONV3': 1, 'o_k_in_CONV3': 5, 'o_x_in_CONV3': 2, 'o_y_in_CONV3': 3, 'o_r_in_CONV3': 4, 'o_s_in_CONV3': 4, 
				't_c_d1_CONV3': 8, 't_c_d2_CONV3': 64, 't_c_d3_CONV3': 1,
				 't_k_d1_CONV3': 8, 't_k_d2_CONV3': 16, 't_k_d3_CONV3': 1, 
				't_x_d1_CONV3': 11, 't_x_d2_CONV3': 1, 't_x_d3_CONV3': 1, 
				't_y_d1_CONV3': 2, 't_y_d2_CONV3': 55, 't_y_d3_CONV3': 1, 

				'o_c_out_CONV4': 1, 'o_k_out_CONV4': 3, 'o_x_out_CONV4': 3, 'o_y_out_CONV4': 1, 'o_r_out_CONV4': 2, 'o_s_out_CONV4': 5, 
				'o_c_mid_CONV4': 2, 'o_k_mid_CONV4': 5, 'o_x_mid_CONV4': 3, 'o_y_mid_CONV4': 6, 'o_r_mid_CONV4': 1, 'o_s_mid_CONV4': 1,
				 'o_c_in_CONV4': 4, 'o_k_in_CONV4': 4, 'o_x_in_CONV4': 4, 'o_y_in_CONV4': 5, 'o_r_in_CONV4': 2, 'o_s_in_CONV4': 1, 
				't_c_d1_CONV4': 1, 't_c_d2_CONV4': 32, 't_c_d3_CONV4': 1, 
				't_k_d1_CONV4': 8, 't_k_d2_CONV4': 4, 't_k_d3_CONV4': 1, 
				't_x_d1_CONV4': 5, 't_x_d2_CONV4': 1, 't_x_d3_CONV4': 1, 
				't_y_d1_CONV4': 5, 't_y_d2_CONV4': 2, 't_y_d3_CONV4': 1, 

				'o_c_out_CONV5': 5, 'o_k_out_CONV5': 6, 'o_x_out_CONV5': 1, 'o_y_out_CONV5': 1, 'o_r_out_CONV5': 4, 'o_s_out_CONV5': 1,
				 'o_c_mid_CONV5': 3, 'o_k_mid_CONV5': 4, 'o_x_mid_CONV5': 4, 'o_y_mid_CONV5': 5, 'o_r_mid_CONV5': 6, 'o_s_mid_CONV5': 6, 
				'o_c_in_CONV5': 5, 'o_k_in_CONV5': 5, 'o_x_in_CONV5': 5, 'o_y_in_CONV5': 1, 'o_r_in_CONV5': 5, 'o_s_in_CONV5': 3, 
				't_c_d1_CONV5': 2, 't_c_d2_CONV5': 16, 't_c_d3_CONV5': 1, 
				't_k_d1_CONV5': 256, 't_k_d2_CONV5': 256, 't_k_d3_CONV5': 1, 
				't_x_d1_CONV5': 6, 't_x_d2_CONV5': 9, 't_x_d3_CONV5': 1, 
				't_y_d1_CONV5': 2, 't_y_d2_CONV5': 18, 't_y_d3_CONV5': 1, 

				'o_c_out_CONV6': 1, 'o_k_out_CONV6': 5, 'o_x_out_CONV6': 4, 'o_y_out_CONV6': 5, 'o_r_out_CONV6': 2, 'o_s_out_CONV6': 4, 
				'o_c_mid_CONV6': 5, 'o_k_mid_CONV6': 5, 'o_x_mid_CONV6': 2, 'o_y_mid_CONV6': 3, 'o_r_mid_CONV6': 1, 'o_s_mid_CONV6': 2, 
				'o_c_in_CONV6': 6, 'o_k_in_CONV6': 1, 'o_x_in_CONV6': 5, 'o_y_in_CONV6': 4, 'o_r_in_CONV6': 4, 'o_s_in_CONV6': 6, 
				't_c_d1_CONV6': 32, 't_c_d2_CONV6': 256, 't_c_d3_CONV6': 1, 
				't_k_d1_CONV6': 256, 't_k_d2_CONV6': 8, 't_k_d3_CONV6': 1, 
				't_x_d1_CONV6': 1, 't_x_d2_CONV6': 2, 't_x_d3_CONV6': 1, 
				't_y_d1_CONV6': 1, 't_y_d2_CONV6': 9, 't_y_d3_CONV6': 1, 

				'o_c_out_CONV7': 3, 'o_k_out_CONV7': 2, 'o_x_out_CONV7': 2, 'o_y_out_CONV7': 1, 'o_r_out_CONV7': 3, 'o_s_out_CONV7': 5, 
				'o_c_mid_CONV7': 4, 'o_k_mid_CONV7': 3, 'o_x_mid_CONV7': 5, 'o_y_mid_CONV7': 2, 'o_r_mid_CONV7': 6, 'o_s_mid_CONV7': 3, 
				'o_c_in_CONV7': 3, 'o_k_in_CONV7': 2, 'o_x_in_CONV7': 3, 'o_y_in_CONV7': 2, 'o_r_in_CONV7': 5, 'o_s_in_CONV7': 2, 
				't_c_d1_CONV7': 128, 't_c_d2_CONV7': 4, 't_c_d3_CONV7': 1, 
				't_k_d1_CONV7': 128, 't_k_d2_CONV7': 64, 't_k_d3_CONV7': 1, 
				't_x_d1_CONV7': 1, 't_x_d2_CONV7': 6, 't_x_d3_CONV7': 1, 
				't_y_d1_CONV7': 3, 't_y_d2_CONV7': 54, 't_y_d3_CONV7': 1, 

				'o_c_out_CONV8': 5, 'o_k_out_CONV8': 5, 'o_x_out_CONV8': 3, 'o_y_out_CONV8': 2, 'o_r_out_CONV8': 2, 'o_s_out_CONV8': 3, 
				'o_c_mid_CONV8': 5, 'o_k_mid_CONV8': 6, 'o_x_mid_CONV8': 4, 'o_y_mid_CONV8': 5, 'o_r_mid_CONV8': 1, 'o_s_mid_CONV8': 3, 
				'o_c_in_CONV8': 1, 'o_k_in_CONV8': 2, 'o_x_in_CONV8': 1, 'o_y_in_CONV8': 5, 'o_r_in_CONV8': 6, 'o_s_in_CONV8': 6, 
				't_c_d1_CONV8': 64, 't_c_d2_CONV8': 2, 't_c_d3_CONV8': 1, 
				't_k_d1_CONV8': 256, 't_k_d2_CONV8': 64, 't_k_d3_CONV8': 1, 
				't_x_d1_CONV8': 2, 't_x_d2_CONV8': 1, 't_x_d3_CONV8': 1,
				 't_y_d1_CONV8': 2, 't_y_d2_CONV8': 26, 't_y_d3_CONV8': 1, 

				'o_c_out_CONV9': 2, 'o_k_out_CONV9': 6, 'o_x_out_CONV9': 3, 'o_y_out_CONV9': 3, 'o_r_out_CONV9': 4, 'o_s_out_CONV9': 6, 
				'o_c_mid_CONV9': 4, 'o_k_mid_CONV9': 5, 'o_x_mid_CONV9': 4, 'o_y_mid_CONV9': 3, 'o_r_mid_CONV9': 2, 'o_s_mid_CONV9': 5, 
				'o_c_in_CONV9': 2, 'o_k_in_CONV9': 6, 'o_x_in_CONV9': 1, 'o_y_in_CONV9': 2, 'o_r_in_CONV9': 2, 'o_s_in_CONV9': 1, 
				't_c_d1_CONV9': 32, 't_c_d2_CONV9': 2, 't_c_d3_CONV9': 1, 
				't_k_d1_CONV9': 16, 't_k_d2_CONV9': 256, 't_k_d3_CONV9': 1, 
				't_x_d1_CONV9': 1, 't_x_d2_CONV9': 2, 't_x_d3_CONV9': 1, 
				't_y_d1_CONV9': 1, 't_y_d2_CONV9': 2, 't_y_d3_CONV9': 1, 

				'o_c_out_CONV10': 2, 'o_k_out_CONV10': 4, 'o_x_out_CONV10': 1, 'o_y_out_CONV10': 2, 'o_r_out_CONV10': 5, 'o_s_out_CONV10': 5, 
				'o_c_mid_CONV10': 4, 'o_k_mid_CONV10': 6, 'o_x_mid_CONV10': 4, 'o_y_mid_CONV10': 1, 'o_r_mid_CONV10': 6, 'o_s_mid_CONV10': 6, 
				'o_c_in_CONV10': 2, 'o_k_in_CONV10': 3, 'o_x_in_CONV10': 5, 'o_y_in_CONV10': 5, 'o_r_in_CONV10': 3, 'o_s_in_CONV10': 4, 
				't_c_d1_CONV10': 128, 't_c_d2_CONV10': 2, 't_c_d3_CONV10': 1,
				 't_k_d1_CONV10': 512, 't_k_d2_CONV10': 128, 't_k_d3_CONV10': 1, 
				't_x_d1_CONV10': 2, 't_x_d2_CONV10': 1, 't_x_d3_CONV10': 1, 
				't_y_d1_CONV10': 1, 't_y_d2_CONV10': 13, 't_y_d3_CONV10': 1, 

				'o_c_out_CONV11': 4, 'o_k_out_CONV11': 1, 'o_x_out_CONV11': 3, 'o_y_out_CONV11': 2, 'o_r_out_CONV11': 2, 'o_s_out_CONV11': 3,
				 'o_c_mid_CONV11': 4, 'o_k_mid_CONV11': 1, 'o_x_mid_CONV11': 2, 'o_y_mid_CONV11': 3, 'o_r_mid_CONV11': 3, 'o_s_mid_CONV11': 5, 
				'o_c_in_CONV11': 4, 'o_k_in_CONV11': 3, 'o_x_in_CONV11': 2, 'o_y_in_CONV11': 1, 'o_r_in_CONV11': 1, 'o_s_in_CONV11': 3, 
				't_c_d1_CONV11': 256, 't_c_d2_CONV11': 2, 't_c_d3_CONV11': 1, 
				't_k_d1_CONV11': 16, 't_k_d2_CONV11': 16, 't_k_d3_CONV11': 1, 
				't_x_d1_CONV11': 4, 't_x_d2_CONV11': 3, 't_x_d3_CONV11': 1, 
				't_y_d1_CONV11': 12, 't_y_d2_CONV11': 12, 't_y_d3_CONV11': 1, 

				'o_c_out_CONV12': 6, 'o_k_out_CONV12': 5, 'o_x_out_CONV12': 5, 'o_y_out_CONV12': 2, 'o_r_out_CONV12': 2, 'o_s_out_CONV12': 4, 
				'o_c_mid_CONV12': 2, 'o_k_mid_CONV12': 5, 'o_x_mid_CONV12': 1, 'o_y_mid_CONV12': 5, 'o_r_mid_CONV12': 2, 'o_s_mid_CONV12': 3,
				 'o_c_in_CONV12': 2, 'o_k_in_CONV12': 2, 'o_x_in_CONV12': 3, 'o_y_in_CONV12': 5, 'o_r_in_CONV12': 2, 'o_s_in_CONV12': 3,
				 't_c_d1_CONV12': 1, 't_c_d2_CONV12': 128, 't_c_d3_CONV12': 1,
				 't_k_d1_CONV12': 8, 't_k_d2_CONV12': 256, 't_k_d3_CONV12': 1,
				 't_x_d1_CONV12': 1, 't_x_d2_CONV12': 2, 't_x_d3_CONV12': 1, 
				't_y_d1_CONV12': 4, 't_y_d2_CONV12': 6, 't_y_d3_CONV12': 1, 

				'o_c_out_CONV13': 5, 'o_k_out_CONV13': 1, 'o_x_out_CONV13': 4, 'o_y_out_CONV13': 2, 'o_r_out_CONV13': 5, 'o_s_out_CONV13': 1, 
				'o_c_mid_CONV13': 1, 'o_k_mid_CONV13': 3, 'o_x_mid_CONV13': 3, 'o_y_mid_CONV13': 5, 'o_r_mid_CONV13': 5, 'o_s_mid_CONV13': 2, 
				'o_c_in_CONV13': 4, 'o_k_in_CONV13': 1, 'o_x_in_CONV13': 3, 'o_y_in_CONV13': 6, 'o_r_in_CONV13': 5, 'o_s_in_CONV13': 3, 
				't_c_d1_CONV13': 4, 't_c_d2_CONV13': 32, 't_c_d3_CONV13': 1, 
				't_k_d1_CONV13': 32, 't_k_d2_CONV13': 128, 't_k_d3_CONV13': 1, 
				't_x_d1_CONV13': 4, 't_x_d2_CONV13': 1, 't_x_d3_CONV13': 1, 
				't_y_d1_CONV13': 1, 't_y_d2_CONV13': 6, 't_y_d3_CONV13': 1
			}			

	else:
		pass	

	metrics = maestro.evaluate(status_example)