import trace
import log
import Queue
import numpy as np
import matplotlib.pyplot as plt


def main():

	def speed(x, p, w):
		return 1.0/(x[0]*40/w + x[1] + x[2]*w/p + x[3]*w + x[4]*p)

	x = [1.02, 2.78, 4.92, 0, 0.02]

	for i in range(1,16):
		print speed(x, i, i)











if __name__ == "__main__":
	main()