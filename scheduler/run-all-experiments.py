import os
import time
import datetime


os.system("mkdir -p ~/exp/; rm -rf ~/exp/*; cp *.py *.txt ~/exp/")
backup_dir = "/home/net/exp" + "-" + datetime.datetime.today().strftime('%Y%m%d_%H%M%S') + "/"
os.system("mkdir -p " + backup_dir)
for i in range(4,12):
	os.system("cd ~/exp/ && python experiment.py " + str(i))
print "finishing testing all..."
os.system("cd ~/exp/ && find . -name '*.txt' -exec cp --parents \{\} " + backup_dir + " \;")
print "finishing backup, dir:", backup_dir