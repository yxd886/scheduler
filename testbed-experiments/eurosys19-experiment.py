import os
import sys
import time
import ast
from k8s_job import K8SJob
import trace
import jobrepo
import log


NODE_LIST=["10.28.1.18", "10.28.1.19", "10.28.1.20", "10.28.1.21", "10.28.1.22", "10.28.1.23", ]
MEM_PER_NODE = 48
GPU_PER_NODE = 2
CPU_PER_NODE = 16

def measure(testfile):
    global logger

    JOB_EPOCHS = [28, 89, 12, 19, 42, 6]
    job_id = 0
    cwd = os.getcwd() + '/'
    stats = []  # element format (#ps, #worker, speed, cpu)
    txt = testfile + ".result.txt"
    if os.path.isfile(txt):  # back up
        time_str = str(int(time.time()))
        os.system("mkdir -p ./results")
        fn = './results/' + time_str + '.' + txt
        os.system('cp ' + txt + ' ' + fn)
    f = open(txt, 'w')  # clear txt
    f.close()

    traces = trace.Trace(None).get_trace()
    ts = 0
    id = 1
    while True:
        jobs = traces[ts] # only consider one trace so far
        k8s_jobs = []
        for job in jobs:
            model = job.model
            for i in range(len(jobrepo.job_repos)):
                _type, _model = jobrepo.job_repos[i]
                if model == _model:
                    k8s_job = Job(id, _type, _model, i, cwd, logger)
                    jobrepo.set_config(k8s_job)
                    k8s_job.num_epochs = job.num_epochs # here specify total number of trained epochs
                    k8s_job.arrv_time = time.time()
                    k8s_job.arrival_slot = ts
                    k8s_job.num_worker = 0
                    k8s_job.num_ps = 0
                    k8s_job.worker_placement = []
                    k8s_job.ps_placement = []
                    k8s_jobs.append(k8s_job)
                    break
        assert len(k8s_jobs) == len(jobs)




                    job_id = id
                    model_id = id
                    (type, model) = jobrepo.job_repos[model_id]
                    job = Job(job_id, type, model, model_id, cwd, logger)
                    jobrepo.set_config(job)
                    job.num_worker = 0
                    job.num_ps = 0
                    job.worker_placement = []
                    job.ps_placement = []
                    jobs[job.id] = job
                    arrivals[ts].append(job_id)
                    arrival_ts[job_id] = ts
                if arrival_ts[id] == ts:
                    if task % 2 == 0:
                            jobs[id].num_worker += 1
                            jobs[id].worker_placement.append(NODE_LIST[machine])
                    else:
                        jobs[id].num_ps += 1
                        jobs[id].ps_placement.append(NODE_LIST[machine])
                if ts not in runnings.keys():
                    runnings[ts] = set()
                runnings[ts].add(id)

    print arrivals, arrival_ts
    for job in jobs.values():
        print job.id, job.num_ps, job.ps_placement, job.num_worker, job.worker_placement

    print runnings

    kill_jobs = dict() # ts: job id set
    for ts in runnings:
        if ts-1 not in runnings:
            continue
        prev_jobs = runnings[ts-1]
        for id in prev_jobs:
            if id not in runnings[ts]:
                if ts not in kill_jobs:
                    kill_jobs[ts] = set()
                kill_jobs[ts].add(id)

    print kill_jobs

    tic = time.time()
    running_jobs = []
    for ts in range(1,1000):
        if ts not in arrivals and ts <max(arrivals):
            continue
        logger.info("timeslot: " + str(ts) + " time passed: " + str((time.time()-tic)/60) + " minutes")
        # kill jobs first
        if ts in kill_jobs:
            logger.info("timeslot: " + str(ts) + " kill jobs: " + str(kill_jobs[ts]))
            for id in kill_jobs[ts]:
                try:
                    jobs[id].delete(True)
                    jobs[id].end_time = time.time()
                    running_jobs.remove(id)
                except Exception as e:
                    logger.info("kill job error! " + str(e))

        if ts in arrivals:
            # start arrival jobs
            logger.info("timeslot: " + str(ts) + " arrival jobs: " + str(arrivals[ts]))
            for id in arrivals[ts]:
                jobs[id].start()
                jobs[id].arrival_time = time.time()
                running_jobs.append(id)

        # monitor job progress
        counter = 0
        while True:
            logger.info("counter: " + str(counter))
            try:
                time.sleep(60)
            except:
                logger.info("detect Ctrl+C, exit...")
                for job in running_jobs:
                    jobs[job].delete(True)
                exit(0)
            counter += 1

            for job in list(running_jobs):
                job = jobs[job]
                try:
                    speed_list = job.get_speed()
                    if sum(speed_list) == 0:
                        logger.info("speed is 0, continue")
                        continue
                except Exception as e:
                    logger.info("get speed error!" + str(e))
                    continue
                try:
                    progress_list = job.get_progress()
                except Exception as e:
                    logger.info("get progress error!" + str(e))
                    continue
                try:
                    (ps_metrics, worker_metrics) = job.get_metrics()
                except Exception as e:
                    logger.info("get metrics error!" + str(e))
                    continue

                # compute cpu usage difference
                ps_cpu_usage_list = []
                for metrics in ps_metrics:
                    ps_cpu_usage_list.append(metrics['cpu/usage_rate'] / 1000.0)
                worker_cpu_usage_list = []
                for metrics in worker_metrics:
                    worker_cpu_usage_list.append(metrics['cpu/usage_rate'] / 1000.0)

                progress = 0
                for epoch, batch in progress_list:
                    progress += epoch
                    progress += 1.0*batch/job.epoch_size

                speed = (sum(speed_list)/job.tot_batch_size)/job.epoch_size
                est_rct = (JOB_EPOCHS[job.id] - progress)/speed
                est_jct = est_rct + (time.time()- job.arrival_time)

                logger.info("job name: " + job.name + " model name: " + job.model_name + ", kv_store: " + job.kv_store + \
                            ", batch_size: " + str(job.tot_batch_size) + \
                            ", num_ps: " + str(job.num_ps) + ", num_worker: " + str(job.num_worker) + \
                            ", progress_list: " + str(progress_list) + \
                            ", total progress: " + str(progress) + \
                            ", total num_epochs: " + str(JOB_EPOCHS[job.id]) + \
                            ", speed_list: " + str(speed_list) + ", sum_speed (samples/second): " + str(sum(speed_list)) + \
                            ", sum_speed(batches/second): " + str(sum(speed_list) / int(job.tot_batch_size)) + \
                            ", est_rct: " + str(est_rct/60) + " minutes" + \
                            ", est_jct: " + str(est_jct/60) + " minutes" + \
                            ", ps cpu usage: " + str(ps_cpu_usage_list) + \
                            ", worker cpu usage: " + str(worker_cpu_usage_list)
                            )

                if progress >= JOB_EPOCHS[job.id]:
                    logger.info("job " + str(job.name) + " finished!")
                    job.end_time = time.time()
                    job.delete(True)
                    running_jobs.remove(job.id)

                if ts > max(arrivals.keys()):  # no jobs arrive any more
                    logger.info("job " + str(job.name) + " is forced to be finished!")
                    job.end_time = time.time() + est_rct
                    job.delete(True)
                    running_jobs.remove(job.id)

            if len(running_jobs) == 0:
                logger.info("All jobs are finished!")
                jct = sum(job.end_time - job.arrival_time for job in jobs.values())/len(jobs)
                makespan = max(job.end_time for job in jobs.values()) - min(job.arrival_time for job in jobs.values())
                for job in jobs.values():
                    logger.info("job id: " + str(job.id) + " arrival_time: " + str(job.arrival_time) + " completion time: " + str(job.end_time))
                logger.info("JCT: " + str(jct) + " Makespan: " + str(makespan))
                with open(txt,'w') as f:
                    f.write("JCT: " + str(jct) + " Makespan: " + str(makespan) + '\n')

                exit(0)
            logger.info("\n")
            if counter == 20:  # a timeslot is 20 minutes
                break




def prepare_env():
    logger.info("clear all existing jobs...")
    os.system("kubectl delete jobs --all")


def run():
    global logger

    logger = log.getLogger("eurosys19-experiment")
    logger.info("AFTER UPDATING SCRIPTS IN CONTAINER, REBUILD THE IMAGE!!!")
    prepare_env()
    measure()

    logger.info("Experiment is over!")
    

if __name__ == '__main__':
    run()
