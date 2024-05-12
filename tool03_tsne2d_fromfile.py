import pandas
import sys
sys.path.append("./util/")
from space import tsne2D_fromfile

goal = "latency"
iindex = 13
#member_list = ["brave_phd", "discreet_phd", "brave_master", "discreet_master", "brave_master", "discreet_master", "brave_tutor", "discreet_tutor"]
#member_list = ["brave_phd", "discreet_phd", "brave_master", "discreet_master", "brave_tutor", "discreet_tutor"]
#member_list = ["discreet_phd", "discreet_phd", "discreet_master", "discreet_master", "discreet_master", "discreet_master", "discreet_tutor", "discreet_tutor",]
member_list = ["brave_phd", "discreet_phd", "brave_master", "discreet_master", "brave_tutor", "discreet_tutor", "brave_reserve", "discreet_reserve"]
jindex_list = [i for i in range(len(member_list))]

obs_file_list = list()
reward_file_list = list()
for jindex, member in zip(jindex_list, member_list):
	obs_file_list.append("record/detail/obs/CSDSE_obs_{}_i={}_j={}_type={}.csv".format(goal, iindex, jindex, member))
	reward_file_list.append("record/detail/reward/CSDSE_reward_{}_i={}_j={}_type={}.csv".format(goal, iindex, jindex, member))
has_interval = True
interval = 1000

tsne2D_fromfile(obs_file_list, reward_file_list, has_interval, interval)