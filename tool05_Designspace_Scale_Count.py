import pdb
import os
import sys
sys.set_int_max_str_digits(0)

sys.path.append("./util/")
from space import create_space_maestro

def int_to_scientific_count(n):
	s = repr(n)
	head = s[0]
	tail = s[1:3]
	l = len(s)-1
	scientific_expression = {"num":"{}.{}".format(head, tail), "ext":"{}".format(l)} 
	return scientific_expression

if __name__ == '__main__':
	#model_list = {"AlexNet", "VGG16", "MobileNetV2", "MnasNet", "GoogleNet", "ResNet50", "ResNet101", "ResNet152"}
	model_list = {"VGG16", "MobileNetV2", "MnasNet", "ResNet50", "GNMT", "Transformer"}

	for model in model_list:
		print(f"Model: {model}")
		DSE_action_space = create_space_maestro(model)
		lenth = DSE_action_space.get_lenth()
		scale = DSE_action_space.get_scale()
		sci_scale = int_to_scientific_count(scale)	
		print(f"DSE_action_space lenth: {lenth}")
		print(f"DSE_action_space scale: {scale}, {sci_scale}")
		filepath = "./data/corr_table_{}.csv".format(model)
		DSE_action_space.corr_analysis(filepath)
		print(f"\n")
