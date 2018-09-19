echo "Running on net-g4"
dir=`pwd`
ssh net-g4 "mkdir -p $dir"
dest="net-g4:$dir"
scp *.py $dest
ssh -f net@net-g4 "sudo pkill -9 python; sudo pkill -9 tensorboard; sleep 3; cd $dir && nohup python train.py &"