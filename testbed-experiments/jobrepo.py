import os
import sys
import time
import datetime
import math
import threading
import random

'''
record time for each epoch when only 1 worker and 1 ps with the following specified resources.

number of servers: 6
server configuration: 8 CPU cores, 40GB memory, 2 GPUs, inter-server bandwidth 6Gbps, intra-server bandwidth 30Gbps. 

cut down the dataset to make each epoch within 5 minutes
time slot: 20-30 min

The table below shows the workload details in the experiment.
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                            |         speed (samples/sec)         |
model            dataset    |   dist-inter   dist-intra    local  |   number of batches    batch size    epoch time (min)   model size (MB)  computation (msec)
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
resnet-50        imagenet           20.5       40.9        67.2               115               32             3                  102.2              449
vgg-16           imagenet           11         17.5        59.8               115               32             5.7                553.4              535
resnext-110      cifar10            374        381.2       567.2              390               128            2.3                6.92               226
inception-bn     caltech256         116        149.9       157                120               128            2.2                42.1               815
seq2seq          wmt17              169        231.5       856                780               64             4.9                36.5               75
ctc              mr                 51         68.0        85.5               193               50             3.2                24                 585
dssm             text8              405        388.9       450.2              349               256            3.7                6                  567
wlm              ptb                80         161.2       1036.4             165               160            5.5                19.2               154
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

dist-inter bandwidth for each worker:  735, 1864, 476, 1167, 961, 486, 737, 83 Mbps
dist_intra bandwidth for each worker:  2452, 3422, 504, 8657, 1449, 1277, 525, 183 Mbps

Maximum speed                             between two k8s containers                                          between two processes(single thead vs multiple threads)
On the same server:                           34 | 70 Gbps (3 threads)                                                           48.0 | 161 Gbps 
Across servers:         5.7 | 9.7 Gbps (2 threads, more leads to bad throughput, drop quickly when >=4 threads)                  19.7 | 47.0 Gbps

'''

# MLP-mnist, AlexNet-Imagenet are also ready as examples
# use DS2-Librispeech when configuration is fixed and more examples are needed.
job_repos = [('experiment-imagenet', 'resnet-50'), ('experiment-imagenet', 'vgg-16'),
             ('experiment-cifar10', 'resnext-110'),('experiment-caltech256', 'inception-bn'),
             ('experiment-wmt17', 'seq2seq'),('experiment-mr', 'cnn-text-classification'),
             ('experiment-text8', 'dssm'), ('experiment-ptb', 'wlm')]

DEFAULT_NUM_PS = 1
DEFAULT_NUM_WORKER = 1

def set_config(job):
    is_member = False
    for item in job_repos:
        if job.type == item[0] and job.model_name == item[1]:
            is_member = True

    if is_member == False:
        raise RuntimeError

    if 'resnet-50' in job.model_name:
        _set_resnet50_job(job)
    elif 'vgg-16' in job.model_name:
        _set_vgg16_job(job)
    elif 'resnext-110' in job.model_name:
        _set_resnext110_job(job)
    elif 'inception-bn' in job.model_name:
        _set_inceptionbn_job(job)
    elif 'dssm' in job.model_name:
        _set_dssm_job(job)
    elif 'seq2seq' in job.model_name:
        _set_seq2seq_job(job)
    elif 'cnn-text-classification' in job.model_name:
        _set_ctc_job(job)
    elif 'wlm' in job.model_name:
        _set_wlm_job(job)
    elif 'ds2' in job.model_name:
        _set_ds2_job(job)
    else:
        raise RuntimeError


'''
ResNet-50_ImageNet
'''
def _set_resnet50_job(job):
    num_ps = DEFAULT_NUM_PS
    num_worker = DEFAULT_NUM_WORKER
    ps_cpu = 3
    ps_mem = 4
    ps_bw = 0
    worker_cpu = 2
    worker_mem = 5
    worker_gpu = 1
    worker_bw = 0

    job.set_ps_resources(num_ps, ps_cpu, ps_mem, ps_bw)
    job.set_worker_resources(num_worker, worker_cpu, worker_mem, worker_bw, worker_gpu)

    image = 'yhpeng/k8s-mxnet-gpu-experiment'
    script = '/init.sh'
    # must end with /, save everything including training data, validation data, training log and training model into this dir
    work_dir = '/mxnet/example/image-classification/data/'
    host_workdir_prefix = '/data/k8s-workdir/experiment/'
    job.set_container(image, script, work_dir, host_workdir_prefix)

    prog = 'python train_imagenet.py --network resnet --num-layers 50 --disp-batches 5 --num-epochs 100 --data-train /data/imagenet-train.rec'
    kv_store = 'dist_sync'
    prog += ' --kv-store ' + kv_store
    if worker_gpu > 0:
        prog += " --gpus" + " " + ",".join([str(i) for i in range(int(worker_gpu))])

    job.set_train(prog=prog, batch_size=32, kv_store=kv_store, scale_bs=True, num_examples=3680, num_epochs=10)
    hdfs_data = ['/k8s-mxnet/imagenet/imagenet-train.rec']
    data_dir = '/data/'
    host_data_dir = '/data/mxnet-data/imagenet/'
    job.set_data(hdfs_data=hdfs_data, data_dir=data_dir, host_data_dir=host_data_dir, data_mounted=True)
    job.set_mxnet(kv_store_big_array_bound=1000 * 1000, ps_verbose='')


'''
VGG-16_ImageNet
'''
def _set_vgg16_job(job):
    num_ps = DEFAULT_NUM_PS
    num_worker = DEFAULT_NUM_WORKER
    ps_cpu = 4
    ps_mem = 5
    ps_bw = 0
    worker_cpu = 2
    worker_mem = 5
    worker_gpu = 1
    worker_bw = 0

    job.set_ps_resources(num_ps, ps_cpu, ps_mem, ps_bw)
    job.set_worker_resources(num_worker, worker_cpu, worker_mem, worker_bw, worker_gpu)

    image = 'yhpeng/k8s-mxnet-gpu-experiment'
    script = '/init.sh'
    # must end with /, save everything including training data, validation data, training log and training model into this dir
    work_dir = '/mxnet/example/image-classification/data/'
    host_workdir_prefix = '/data/k8s-workdir/experiment/'
    job.set_container(image, script, work_dir, host_workdir_prefix)

    prog = 'python train_imagenet.py --network vgg --num-layers 16 --disp-batches 2 --num-epochs 100 --data-train /data/imagenet-train.rec'
    kv_store = 'dist_sync'
    prog += ' --kv-store ' + kv_store
    if worker_gpu > 0:
        prog += " --gpus" + " " + ",".join([str(i) for i in range(int(worker_gpu))])
    job.set_train(prog=prog, batch_size=32, kv_store=kv_store, scale_bs=True, num_examples=3680, num_epochs=10)
    hdfs_data = ['/k8s-mxnet/imagenet/imagenet-train.rec']
    data_dir = '/data/'
    host_data_dir = '/data/mxnet-data/imagenet/'
    job.set_data(hdfs_data=hdfs_data, data_dir=data_dir, host_data_dir=host_data_dir, data_mounted=True)
    job.set_mxnet(kv_store_big_array_bound=1000 * 1000, ps_verbose='')


'''
ResNext-110_Cifar10
'''
def _set_resnext110_job(job):
    num_ps = DEFAULT_NUM_PS
    num_worker = DEFAULT_NUM_WORKER
    ps_cpu = 3
    ps_mem = 3
    ps_bw = 0
    worker_cpu = 2
    worker_mem = 4
    worker_gpu = 1
    worker_bw = 0

    job.set_ps_resources(num_ps, ps_cpu, ps_mem, ps_bw)
    job.set_worker_resources(num_worker, worker_cpu, worker_mem, worker_bw, worker_gpu)

    image = 'yhpeng/k8s-mxnet-gpu-experiment'
    script = '/init.sh'
    # must end with /, save everything including training data, validation data, training log and training model into this dir
    work_dir = '/mxnet/example/image-classification/data/'
    host_workdir_prefix = '/data/k8s-workdir/experiment/'
    job.set_container(image, script, work_dir, host_workdir_prefix)

    prog = 'python train_cifar10.py --network resnext --num-layers 110 --disp-batches 25 \
    --num-epochs 100 --data-train /data/cifar10-train.rec --data-val /data/cifar10-val.rec'
    kv_store = 'dist_sync'
    prog += ' --kv-store ' + kv_store
    if worker_gpu > 0:
        prog += " --gpus" + " " + ",".join([str(i) for i in range(int(worker_gpu))])
    job.set_train(prog=prog, batch_size=128, kv_store=kv_store, scale_bs=True, num_examples=50000, num_epochs=10)
    hdfs_data = ['/k8s-mxnet/cifar10/cifar10-train.rec']
    data_dir = '/data/'
    host_data_dir = '/data/mxnet-data/cifar10/'
    job.set_data(hdfs_data=hdfs_data, data_dir=data_dir, host_data_dir=host_data_dir, data_mounted=True)
    job.set_mxnet(kv_store_big_array_bound=1000 * 1000, ps_verbose='')


'''
Inception-BN_Caltech256
'''
def _set_inceptionbn_job(job):
    num_ps = DEFAULT_NUM_PS
    num_worker = DEFAULT_NUM_WORKER
    ps_cpu = 3
    ps_mem = 4
    ps_bw = 0
    worker_cpu = 2
    worker_mem = 5
    worker_gpu = 1
    worker_bw = 0

    job.set_ps_resources(num_ps, ps_cpu, ps_mem, ps_bw)
    job.set_worker_resources(num_worker, worker_cpu, worker_mem, worker_bw, worker_gpu)

    image = 'yhpeng/k8s-mxnet-gpu-experiment'
    script = '/init.sh'

    # must end with /, save everything including training data, validation data, training log and training model into this dir
    work_dir = '/mxnet/example/image-classification/data/'
    host_workdir_prefix = '/data/k8s-workdir/experiment/'
    job.set_container(image, script, work_dir, host_workdir_prefix)

    prog = 'python train_imagenet.py --network inception-bn --disp-batches 5 --num-epochs 100 \
    --data-train /data/caltech256-train.rec --num-classes 256 --num-examples 15240'
    kv_store = 'dist_sync'
    prog += ' --kv-store ' + kv_store
    if worker_gpu > 0:
        prog += " --gpus" + " " + ",".join([str(i) for i in range(int(worker_gpu))])
    job.set_train(prog=prog, batch_size=128, kv_store=kv_store, scale_bs=True, num_examples=15240, num_epochs=10)
    hdfs_data = ['/k8s-mxnet/caltech256/caltech256-train.rec']
    data_dir = '/data/'
    host_data_dir = '/data/mxnet-data/caltech256/'
    job.set_data(hdfs_data=hdfs_data, data_dir=data_dir, host_data_dir=host_data_dir, data_mounted=True)
    job.set_mxnet(kv_store_big_array_bound=1000 * 1000, ps_verbose='')


'''
DSSM_text8
'''
def _set_dssm_job(job):
    num_ps = DEFAULT_NUM_PS
    num_worker = DEFAULT_NUM_WORKER
    ps_cpu = 1
    ps_mem = 2
    ps_bw = 0
    worker_cpu = 4
    worker_mem = 3
    worker_gpu = 0
    worker_bw = 0
    job.set_ps_resources(num_ps, ps_cpu, ps_mem, ps_bw)
    job.set_worker_resources(num_worker, worker_cpu, worker_mem, worker_bw, worker_gpu)

    image = 'yhpeng/k8s-mxnet-gpu-experiment'
    script = '/init.sh'

    # must end with /, save everything including training data, validation data, training log and training model into this dir
    # make sure to create work_dir in image
    work_dir = '/mxnet/example/nce-loss/data/'
    host_workdir_prefix = '/data/k8s-workdir/experiment/'
    job.set_container(image, script, work_dir, host_workdir_prefix)

    prog = 'python wordvec_subwords_dist.py --num-epochs 100 --disp-batches 20'
    kv_store = 'dist_sync'
    prog += ' --kv-store ' + kv_store
    if worker_gpu > 0:
        prog += " --gpus" + " " + ",".join([str(i) for i in range(int(worker_gpu))])
    job.set_train(prog=prog, batch_size=256, kv_store=kv_store, scale_bs=True, num_examples=89344, num_epochs=10)
    hdfs_data = ['/k8s-mxnet/text8/text8-small']
    data_dir = '/data/'
    host_data_dir = '/data/mxnet-data/text8/'
    job.set_data(hdfs_data=hdfs_data, data_dir=data_dir, host_data_dir=host_data_dir, data_mounted=True)
    job.set_mxnet(kv_store_big_array_bound=1000 * 1000, ps_verbose='')



'''
Seq2Seq_WMT17
'''
def _set_seq2seq_job(job):
    num_ps = DEFAULT_NUM_PS
    num_worker = DEFAULT_NUM_WORKER
    ps_cpu = 3
    ps_mem = 5
    worker_cpu = 2
    worker_mem = 5
    worker_gpu = 1
    ps_bw = 0
    worker_bw = 0

    job.set_ps_resources(num_ps, ps_cpu, ps_mem, ps_bw)
    job.set_worker_resources(num_worker, worker_cpu, worker_mem, worker_bw, worker_gpu)

    image = 'yhpeng/k8s-mxnet-gpu-experiment'
    script = '/init.sh'

    # must end with /, save everything including training data, validation data, training log and training model into this dir
    work_dir = '/mxnet/example/nmt/data/'
    host_workdir_prefix = '/data/k8s-workdir/experiment/'
    job.set_container(image, script, work_dir, host_workdir_prefix)

    # default display frequency 50
    prog = 'python3 -m sockeye.train -s /data/wmt17-train.de -t /data/wmt17-train.en  -vs /data/wmt17-val.de -vt /data/wmt17-val.en \
    --min-num-epochs 100 --max-num-epochs 100 --num-embed 128 --rnn-num-hidden 512 --embed-dropout 0.2 --word-min-count 10 \
    --learning-rate-scheduler-type fixed-rate-inv-sqrt-t -o nmt_model'
    kv_store = 'dist_sync'
    prog += ' --kvstore ' + kv_store # not --kv-store
    if worker_gpu > 0:
        prog += " --device-ids" + " " + " ".join([str(i) for i in range(int(worker_gpu))])  # --device-ids -2 means automatically get 2 GPUs.
    else:
        prog += " --use-cpu"
    job.set_train(prog=prog, batch_size=64, kv_store=kv_store, scale_bs=True, num_examples=50000, num_epochs=10)
    hdfs_data = ['/k8s-mxnet/wmt17/wmt17-train.de', '/k8s-mxnet/wmt17/wmt17-train.en',
                 '/k8s-mxnet/wmt17/wmt17-val.de', '/k8s-mxnet/wmt17/wmt17-val.en']
    data_dir = '/data/'
    host_data_dir = '/data/mxnet-data/wmt17/'
    job.set_data(hdfs_data=hdfs_data, data_dir=data_dir, host_data_dir=host_data_dir, data_mounted=True)
    job.set_mxnet(kv_store_big_array_bound=1000 * 1000, ps_verbose='')


'''
ctc_mr
'''
def _set_ctc_job(job):
    num_ps = DEFAULT_NUM_PS
    num_worker = DEFAULT_NUM_WORKER
    ps_cpu = 1
    ps_mem = 2
    worker_cpu = 4
    worker_mem = 2
    worker_gpu = 0
    ps_bw = 0
    worker_bw = 0

    job.set_ps_resources(num_ps, ps_cpu, ps_mem, ps_bw)
    job.set_worker_resources(num_worker, worker_cpu, worker_mem, worker_bw, worker_gpu)

    image = 'yhpeng/k8s-mxnet-gpu-experiment'
    script = '/init.sh'

    # must end with /, save everything including training data, validation data, training log and training model into this dir
    work_dir = '/mxnet/example/cnn_text_classification/data/'
    host_workdir_prefix = '/data/k8s-workdir/experiment/'
    job.set_container(image, script, work_dir, host_workdir_prefix)

    prog = 'python text_cnn.py --disp-batches 5 --num-epochs 100'
    kv_store = 'dist_sync'
    prog += ' --kv-store ' + kv_store
    if worker_gpu > 0:
        prog += " --gpus" + " " + ",".join([str(i) for i in range(int(worker_gpu))])
    job.set_train(prog=prog, batch_size=50, kv_store=kv_store, scale_bs=True, num_examples=9650, num_epochs=10)
    hdfs_data = ['/k8s-mxnet/mr/rt-polarity.neg', '/k8s-mxnet/mr/rt-polarity.pos']
    data_dir = '/data/'
    host_data_dir = '/data/mxnet-data/mr/'
    job.set_data(hdfs_data=hdfs_data, data_dir=data_dir, host_data_dir=host_data_dir, data_mounted=True)
    job.set_mxnet(kv_store_big_array_bound=1000 * 1000, ps_verbose='')


'''
wlm_ptb
'''
def _set_wlm_job(job):
    num_ps = DEFAULT_NUM_PS
    num_worker = DEFAULT_NUM_WORKER
    ps_cpu = 1
    ps_mem = 2
    worker_cpu = 2
    worker_mem = 3
    worker_gpu = 1
    ps_bw = 0
    worker_bw = 0

    job.set_ps_resources(num_ps, ps_cpu, ps_mem, ps_bw)
    job.set_worker_resources(num_worker, worker_cpu, worker_mem, worker_bw, worker_gpu)

    image = 'yhpeng/k8s-mxnet-gpu-experiment'
    script = '/init.sh'

    # must end with /, save everything including training data, validation data, training log and training model into this dir
    work_dir = '/mxnet/example/rnn/word_lm/data/'
    host_workdir_prefix = '/data/k8s-workdir/experiment/'
    job.set_container(image, script, work_dir, host_workdir_prefix)

    prog = 'python train_dist.py --disp-batches 3 --num-epochs 100'
    kv_store = 'dist_sync'
    prog += ' --kv-store ' + kv_store
    if worker_gpu > 0:
        prog += " --gpus" + " " + ",".join([str(i) for i in range(int(worker_gpu))])
    job.set_train(prog=prog, batch_size=160, kv_store=kv_store, scale_bs=True, num_examples=26400, num_epochs=10)
    hdfs_data = ['/k8s-mxnet/ptb/input.txt', '/k8s-mxnet/ptb/ptb.test.txt', '/k8s-mxnet/ptb/ptb.train.txt', '/k8s-mxnet/ptb/ptb.valid.txt']
    data_dir = '/data/'
    host_data_dir = '/data/mxnet-data/ptb/'
    job.set_data(hdfs_data=hdfs_data, data_dir=data_dir, host_data_dir=host_data_dir, data_mounted=True)
    job.set_mxnet(kv_store_big_array_bound=1000 * 1000, ps_verbose='')


'''
ds2_librispeech
'''
def _set_ds2_job(job):
    print "only support experiment with fixed gpu configuration now..."
    raise RuntimeError

    num_ps = DEFAULT_NUM_PS
    num_worker = DEFAULT_NUM_WORKER
    ps_cpu = 4
    ps_mem = 10
    worker_cpu = 2
    worker_mem = 10
    worker_gpu = 1
    ps_bw = 0
    worker_bw = 0

    job.set_ps_resources(num_ps, ps_cpu, ps_mem, ps_bw)
    job.set_worker_resources(num_worker, worker_cpu, worker_mem, worker_bw, worker_gpu)

    image = 'yhpeng/k8s-mxnet-gpu-experiment'
    script = '/init.sh'

    # must end with /, save everything including training data, validation data, training log and training model into this dir
    work_dir = '/mxnet/example/speech_recognition/data/'
    host_workdir_prefix = '/data/k8s-workdir/experiment/'
    job.set_container(image, script, work_dir, host_workdir_prefix)

    prog = 'python main.py --configfile deepspeech.cfg'
    job.set_train(prog=prog, batch_size=14, kv_store=kv_store, scale_bs=True, num_examples=45000, num_epochs=10)
    hdfs_data = ['/k8s-mxnet/librispeech/train_corpus.json', '/k8s-mxnet/librispeech/validation_corpus.json', '/k8s-mxnet/librispeech/test_corpus.json']
    data_dir = '/data/'
    host_data_dir = '/data/mxnet-data/librispeech/'
    job.set_data(hdfs_data=hdfs_data, data_dir=data_dir, host_data_dir=host_data_dir, data_mounted=True)
    job.set_mxnet(kv_store_big_array_bound=1000 * 1000, ps_verbose='')