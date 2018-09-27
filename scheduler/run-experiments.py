import os
import ast

with open("tasks.txt", 'r') as f:
	tasks = ast.literal_eval(f.readline().replace('\n',''))

for task in tasks:
	print "starting task ", task
	os.system("cd ~/exp/ && python experiment.py " + str(task))