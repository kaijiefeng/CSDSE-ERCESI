import os
import pdb
import configparser

class evaluation_stonne():
	def __init__(self, nnmodel):
		self.nnmodel = nnmodel
		self.evaluation_path = '/home/ubuntu/Program/stonne/benchmarks/image_classification/alexnet/execution.py'
		self.archfile_path = '/home/ubuntu/Program/stonne/simulation_files/configured_acrh.cfg'
		self.tilefile_path = '/home/ubuntu/Program/stonne/benchmarks/image_classification/alexnet/tiles/tile_configuration_'

		self.model = []
		if(self.nnmodel == 'alexnet'):
			self.model = [
				{'type':'CONV', 'R':11, 'S':11, 'C':3, 'K':64, 'X':224, 'Y':224, 'strides':4, 'padding':2, 'tile':self.tilefile_path + 'conv1' + '.txt'},
				{'type':'CONV', 'R':5, 'S':5, 'C':64, 'K':192, 'X':27, 'Y':27, 'strides':1, 'padding':2, 'tile':self.tilefile_path + 'conv2' + '.txt'},
				{'type':'CONV', 'R':3, 'S':3, 'C':192, 'K':384, 'X':13, 'Y':13, 'strides':1, 'padding':1, 'tile':self.tilefile_path + 'conv3' + '.txt'},
				{'type':'CONV', 'R':3, 'S':3, 'C':384, 'K':256, 'X':13, 'Y':13, 'strides':1, 'padding':1, 'tile':self.tilefile_path + 'conv4' + '.txt'},
				{'type':'CONV', 'R':3, 'S':3, 'C':256, 'K':256, 'X':13, 'Y':13, 'strides':1, 'padding':1, 'tile':self.tilefile_path + 'conv5' + '.txt'},
				{'type':'FC', 'M':4096, 'K':256*6*6, 'tile':self.tilefile_path + 'fc6' + '.txt'},
				{'type':'FC', 'M':4096, 'K':4096, 'tile':self.tilefile_path + 'fc7' + '.txt'},
				{'type':'FC', 'M':10, 'K':4096, 'tile':self.tilefile_path + 'fc8' + '.txt'},
			]

	def evaluate(self, status):
		#parse the status
		arch = status['arch']
		ms_rows = status['ms_rows']
		ms_cols = status['ms_cols']
		dn_bw = status['dn_bw']
		rn_bw = status['rn_bw']
		tiling = []
		if(self.nnmodel == 'alexnet'): 
			for i, layer in enumerate(self.model):#convs and maxpools
				if(self.model[i]['type'] == 'CONV'):
					#layer index in status is start from 1 rather than 0
					T_R, T_S, T_C, T_K, T_X_, T_Y_ = status['L'+str(i+1)+'_T_R'], status['L'+str(i+1)+'_T_S'], status['L'+str(i+1)+'_T_C'], status['L'+str(i+1)+'_T_K'], status['L'+str(i+1)+'_T_X_'], status['L'+str(i+1)+'_T_Y_']
					tiling.append({'tile_type':'CONV', 'T_R':T_R, 'T_S':T_S, 'T_C':T_C, 'T_G':'1', 'T_K':T_K, 'T_N':'1', 'T_X\'':T_X_, 'T_Y\'':T_Y_, 'tile':layer['tile']})
				elif(self.model[i]['type'] == 'FC'):
					T_M, T_K = status['L'+str(i+1)+"_T_M"], status['L'+str(i+1)+"_T_K"]
					tiling.append({'tile_type':'FC', 'T_S':T_M, 'T_K':T_K, 'T_N':'1', 'tile':layer['tile']})
				else:
					pass

		#construct the descritpion for simulatior
		with open(self.archfile_path, 'w') as fa:
			fa.write('[MSNetwork]'+'\n')
			if(arch == 0): fa.write('type="LINEAR"'+'\n')
			elif(arch == 1): fa.write('type="LINEAR"'+'\n')
			elif(arch == 2): fa.write('type="OS_MESH"'+'\n')
			if(arch == 0 or arch == 1): fa.write('ms_size='+str(ms_rows*ms_cols)+'\n')
			elif(arch == 2): fa.write('ms_rows='+str(ms_rows)+'\n'+'ms_cols='+str(ms_cols)+'\n')

			fa.write('[ReduceNetwork]'+'\n')	
			if(arch == 0): fa.write('type="ASNETWORK"'+'\n')
			elif(arch == 1): fa.write('type="FENETWORK"'+'\n')
			elif(arch == 2): fa.write('type="TEMPORALRN"'+'\n')
			fa.write('accumulation_buffer_enabled=1'+'\n')

			fa.write('[SDMemory]'+'\n')
			fa.write('dn_bw='+str(dn_bw)+'\n')
			fa.write('rn_bw='+str(rn_bw)+'\n')
			if(arch == 0): fa.write('controller_type="MAERI_DENSE_WORKLOAD"'+'\n')
			elif(arch == 1): fa.write('controller_type="SIGMA_SPARSE_GEMM"'+'\n')
			elif(arch == 2): fa.write('controller_type="TPU_OS_DENSE"'+'\n')

		#tile desc
		if(self.nnmodel == 'alexnet'):
			for layer in tiling:
				with open(layer['tile'], 'w') as ft:
					for key, value in layer.items():
						if(key != 'tile'):
							if(key == 'tile_type'): text = key + '=' + '\"' + str(value) + '\"'
							else: text = key + '=' + str(value)
							ft.write(text + '\n')

		#launch the simulator
		cycle, energy = 0, 0
		command = 'python ' + self.evaluation_path
		os.system(command)	

if __name__ == '__main__':
	stonne = evaluation_stonne('alexnet')
	status_example = {
			'arch':2,
			'ms_rows':16,
			'ms_cols':16,
			'dn_bw':32,
			'rn_bw':128,
			'L1_T_R':11,'L1_T_S':11,'L1_T_C':1,'L1_T_K':1,'L1_T_X_':1,'L1_T_Y_':1,
			'L2_T_R':5,'L2_T_S':5,'L2_T_C':1,'L2_T_K':1,'L2_T_X_':1,'L2_T_Y_':1,	
			'L3_T_R':3,'L3_T_S':3,'L3_T_C':1,'L3_T_K':1,'L3_T_X_':1,'L3_T_Y_':1,	
			'L4_T_R':3,'L4_T_S':3,'L4_T_C':1,'L4_T_K':1,'L4_T_X_':1,'L4_T_Y_':1,	
			'L5_T_R':3,'L5_T_S':3,'L5_T_C':1,'L5_T_K':1,'L5_T_X_':1,'L5_T_Y_':1,	
			'L6_T_M':1,'L6_T_K':1,	
			'L7_T_M':1,'L7_T_K':1,	
			'L8_T_M':1,'L8_T_K':1,	
	}
	stonne.evaluate(status_example)


