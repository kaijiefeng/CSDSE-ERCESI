import numpy
import random
import pdb

class replaybuffer():
	def __init__(self, max_size = 10000):
		#initial buffer, for the moment we use list
		self.current_status_buffer = list()
		self.action_buffer = list()
		self.next_status_buffer = list()
		self.reward_buffer = list()
		self.step_buffer = list()
		self.not_done_buffer = list()
		#buffer point
		self.ptr = 0
		self.is_full = False
		self.max_size = max_size

	def add(self, current_status, action, next_status, reward, step, not_done):
		#judge whether buffer is full
		if(len(self.current_status_buffer) < self.max_size):
			self.is_full = False
		else:
			self.is_full = True

		#append new item in list
		if(self.is_full):
			self.current_status_buffer[self.ptr] = current_status
			self.action_buffer[self.ptr] = action
			self.next_status_buffer[self.ptr] = next_status
			self.reward_buffer[self.ptr] = reward
			self.step_buffer[self.ptr] = step
			self.not_done_buffer[self.ptr] = not_done
		else:
			self.current_status_buffer.append(current_status)
			self.action_buffer.append(action)
			self.next_status_buffer.append(next_status)
			self.reward_buffer.append(reward)
			self.step_buffer.append(step)
			self.not_done_buffer.append(not_done)

		#modify point
		self.ptr = (self.ptr + 1) % self.max_size 

	#for that moment (20200910), replaybuffer only return one item
	#in future days, replaybuffer can return a batch items
	def sample(self):
		#randomly create the index
		index = random.randint(0, len(self.current_status_buffer)-1)
		return (
			self.current_status_buffer[index], 
			self.action_buffer[index],
			self.next_status_buffer[index],
			self.reward_buffer[index],
			self.step_buffer[index],
			self.not_done_buffer[index]
		)

