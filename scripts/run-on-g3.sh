echo "Running on net-g3"
dir=`pwd`
ssh net-g3 "mkdir -p $dir"
dest="net-g3:$dir"
scp *.py $dest
ssh -f net@net-g3 "sudo pkill -9 python; sudo pkill -9 tensorboard; sleep 3; cd $dir && nohup python train.py &"
