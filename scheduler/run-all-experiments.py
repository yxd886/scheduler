import os
import time
import datetime


# os.system("mkdir -p ~/exp/; rm -rf ~/exp/*; cp *.py *.txt ~/exp/")
# backup_dir = "/home/net/exp" + "-" + datetime.datetime.today().strftime('%Y%m%d_%H%M%S') + "/"
# os.system("mkdir -p " + backup_dir)
# for i in range(4,12):
# 	os.system("cd ~/exp/ && python experiment.py " + str(i))
# print "finishing testing all..."
# os.system("cd ~/exp/ && find . -name '*.txt' -exec cp --parents \{\} " + backup_dir + " \;")
# print "finishing backup, dir:", backup_dir


servers = ["net-g1", "net-g2", "net-g3", "net-g6", "net-g7", "net-g8"]
tasks = [[3], [2], [4,10,11], [8,16,18], [1,20], [17,22]]
# servers = ["net-g1"]
# tasks = [[3]]
source = "/home/net/DL2"
msg = "make sure " + source + " is latest!"
print msg
msg = "make sure the localhost is net-g1!"
time.sleep(3)

for server in servers:
	print "starting tasks on", server
	dir_name = "DL2-experiment-" + datetime.datetime.today().strftime('%Y%m%d_%H%M%S')
	# copy
	cmd = "scp -r " + source + " " + server + ":~/" + dir_name
	os.system(cmd)
	with open("tasks.txt", 'w') as f:
		task_list = tasks[servers.index(server)]
		f.write(str(task_list)+"\n")
	cmd = "scp tasks.txt " + server + ":~/" + dir_name + "/scheduler/"
	os.system(cmd)
	os.system("rm tasks.txt")
	# start in background
	if server == "net-g1":
		cmd = "'cd " + dir_name + "/scheduler/ && nohup python run-experiments.py > /dev/null 2>&1 &'"
	else:
		cmd = "'pkill -9 python; sleep 1; cd " + dir_name + "/scheduler/ && nohup python run-experiments.py > /dev/null 2>&1 &'"
	cmd = "ssh -f " + server + " " + cmd
	os.system(cmd)
