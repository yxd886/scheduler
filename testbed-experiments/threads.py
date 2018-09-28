import threading

class Threads(object):
	'''
	a class to manage a group of threads
	'''
	def __init__(self, name="threads"):
		self.threads = []
		self.name = name
		self.progress = 0
	
	def add(self, thread):
		'''resource requirements of parameter servers'''
		self.threads.append(thread)

	def start(self):
		for thread in self.threads:
			thread.start()

	def size(self):
		return len(self.threads)

	def wait(self):
		wait_threads = list(self.threads)
		while len(wait_threads) > 0:
			for thread in list(wait_threads):
				thread.join(timeout=1)
				if not thread.isAlive():
					wait_threads.remove(thread)
					self.progress += 1

	def clear(self):
		self.threads = []
		self.progress = 0

	def get_progress(self):
		return self.progress