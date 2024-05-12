import time

class timer():
	def __init__(self):
		self.record_sum = dict()
		self.record_list = dict()
		self.tstart = dict()
		self.tend = dict()
	def start(self, tname):
		self.tstart[tname] = time.perf_counter()
	def end(self, tname):
		self.tend[tname] = time.perf_counter()
		t = self.tend[tname] - self.tstart[tname]
		if(tname not in self.record_sum):
			self.record_sum[tname] = 0
		if(tname not in self.record_list):
			self.record_list[tname] = list()
		self.record_sum[tname] += t
		self.record_list[tname].append(t)
	def get_sum(self, tname):
		if(tname not in self.record_sum):
			tsum = 0
		else:
			tsum = self.record_sum[tname]
		return tsum
	def get_list(self, tname):
		if(tname not in self.record_list):
			tlist = list()
		else:
			tlist = self.record_list[tname]
		return tlist

if __name__ == '__main__':
	timer = timer()
	timer.start("update")
	timer.end("update")
	for index in range(10):
		timer.start("update")
		for jndex in range(100000):
			pass
		timer.end("update")
	print(f"update cost sum:{timer.get_sum('update')}")
	print(f"update cost list:{timer.get_list('update')}")

