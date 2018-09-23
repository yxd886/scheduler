import os
import csv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib
import time
import datetime
import numpy as np
import ast
import yaml

plt.style.use(["seaborn-bright", "single-figure.mplstyle"])


trace_dir = "../trace/ali-v2/"

# gpu 100%
# cpu 100%
# mem MB

class Job(object):
    def __init__(self, id, model, start, end):
        self.id = id
        self.model = model
        self.start = start
        self.end = end
        self.duration = end - start

        self.dist = True
        self.resources = True
        self.num_worker = None
        self.num_ps = None
        self.ps_gpu = None
        self.worker_gpu = None
        self.ps_cpu = None
        self.worker_cpu = None
        self.ps_mem = None
        self.worker_mem = None


    def set_ps_resources(self, num_ps, ps_cpu, ps_mem, ps_gpu):
        self.num_ps = num_ps
        self.ps_cpu = ps_cpu
        self.ps_mem = ps_mem
        self.ps_gpu = ps_gpu

    def set_worker_resources(self, num_worker, worker_cpu, worker_mem, worker_gpu):
        self.num_worker = num_worker
        self.worker_cpu = worker_cpu
        self.worker_mem = worker_mem
        self.worker_gpu = worker_gpu

    @staticmethod
    def format_time(ts):
        return time.localtime(ts)


jobs = []
num_trace_files = 0
for trace_file in sorted(os.listdir(trace_dir)):
    fn = trace_dir + trace_file
    if "job_trace" in trace_file and os.path.isfile(fn):
        pos = fn.index("2018")-1
        if fn[pos] == '.':
            print "Error: some file wrong format", trace_file
            new_fn = fn[:pos] + "_" + fn[pos+1:]
            os.rename(fn, new_fn)
        else:
            num_trace_files += 1
            with open(fn, 'r') as f:
                for line in f:
                    line = line.replace('\n', '')
                    if not line:
                        continue
                    items = line.split(',')
                    id = items[0]
                    model = items[1]
                    start = int(items[2])
                    end = int(items[3])

                    if end == 0: # some jobs have no end time, may with/without ClusterInfo
                        continue

                    job = Job(id, model, start, end)
                    jobs.append(job)
                    resources = ','.join(items[4:])
                    try:
                        start_index = resources.index("ClusterInfo") + len("ClusterInfo") + 4
                    except:
                        job.resources = False
                        continue
                    end_index = len(resources)-1
                    resources = resources[start_index:end_index]
                    if resources:
                        try:
                            resources = ast.literal_eval(resources)
                        except Exception as e:
                            print resources, fn
                            print e
                            exit(1)

                        for key, value in resources.items():
                            if "count" in value:
                                count = value['count']
                            else:
                                count = None
                            if "cpu" in value:
                                cpu = value['cpu']
                            else:
                                cpu = None
                            if "memory" in value:
                                memory = value["memory"]
                            else:
                                memory = None
                            if "gpu" in value:
                                gpu = value["gpu"]
                            else:
                                gpu = None
                            if key == "worker":
                                job.set_worker_resources(count, cpu, memory, gpu)
                            elif key == "ps":
                                job.set_ps_resources(count, cpu, memory, gpu)
                    else:
                        job.dist = False

print "Total # of trace files:", num_trace_files
print "Total # of jobs:", sum([job.dist for job in jobs])
#
# def job_arrival_all():
#     print "Total # of jobs", len(jobs)
#
#     num_jobs_per_day = []
#     jobs_one_day = 0
#     last_day = 0
#     for job in jobs:
#         start = Job.format_time(job.start)
#
#         if start.tm_mday != last_day:
#             last_day = start.tm_mday
#         jobs_one_day += 1
#
#     x = [_ for _ in range(len(tot_jobs))]
#     # plt.plot(x, tot_jobs, 'b-')
#     plt.bar(x, tot_jobs)
#     plt.show()
#
def job_arrival_day():
    starts = []
    for job in jobs:
        start = Job.format_time(job.start)
        if start.tm_year == 2018 and start.tm_mon == 8 and (start.tm_mday >= 20 and start.tm_mday<=26):
            starts.append(job.start)

    dt_obj = datetime.datetime(2018, 8, 20, 0, 0, 0)
    ts = time.mktime(dt_obj.timetuple())
    print ts, min(starts), time.localtime(min(starts)), time.localtime(max(starts))
    # exit()
    relative_starts = [_-ts for _ in starts]
    bins = [_*3600 for _ in range(0,24*7)]
    counts, bin_edges = np.histogram(relative_starts, bins=bins)
    dt_list = [datetime.datetime.fromtimestamp(_+ts) for _ in bins]
    dates = matplotlib.dates.date2num(dt_list)
    fig, ax = plt.subplots()
    ax.plot_date(dates[1:], counts, 'g-')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.ylabel("Job arrival rate")
    plt.gcf().autofmt_xdate()
    plt.show()
    fig.savefig("job_arrival_rate.pdf")


def job_length():
    durations = []
    for job in jobs:
        if job.dist:
            durations.append(job.duration/60)
    # plt.hist(durations, bins=20, cumulative=True, label='CDF', normed=True, histtype='step')
    bins = [_ for _ in range(0, max(durations), max(durations)/1000)]
    counts, bin_edges = np.histogram(durations, bins=bins)
    cdf = np.cumsum(counts, dtype=float)
    cdf /= cdf[-1]
    cdf = np.append(np.array([0.0]), cdf)
    fig, ax = plt.subplots()
    ax.plot(bin_edges, cdf)
    plt.xlabel("Duration (min)")
    plt.ylabel("CDF")
    plt.ylim(0,1)
    plt.xlim(left=1)

    plt.xscale('log')
    plt.gcf().subplots_adjust(bottom=0.2, left=0.2)
    plt.show()
    fig.savefig("job_length_cdf.pdf")


def get_number_of_models():
    models = dict()
    for job in jobs:
        if job.dist:
            if job.model not in models:
                models[job.model] = 1
            else:
                models[job.model] += 1
    print "# of models", len(models)
    print "# of models trained for more than once", sum([m>1 for m in models.values()])
    # plt.hist(models.values(), bins=20)
    # plt.show()

    more_models = dict()
    for k, v in models.items():
        if v > 1:
            more_models[k] = v

    plt.hist(more_models.values(), bins=50)
    plt.show()

def get_gpu_request():
    candidates = []
    for job in jobs:
        start = Job.format_time(job.start)
        if start.tm_year == 2018 and start.tm_mon == 9 and start.tm_mday >= 1 and start.tm_mday <= 5:
            candidates.append(job)
    # calculate Sep. 5
    dt_obj = datetime.datetime(2018, 9, 3, 12, 0, 0) # start from 12:00 noon, so need to adjust xticks
    ts = time.mktime(dt_obj.timetuple())

    for job in candidates:
        if job.worker_gpu and job.num_worker:
            print job.worker_gpu*job.num_worker

    gpus = []
    cpus = []
    mems = []
    for i in range(24*100):
        comp_ts = ts + i*36
        gpu = 0
        cpu = 0
        mem = 0
        for job in candidates:
            if job.start <= comp_ts and job.end > comp_ts:
                # if not job.resources:
                #     if job.num_worker:
                #         job.worker_cpu = 100
                #         job.worker_mem = 1000
                #     if job.num_ps:
                #         job.ps_cpu = 100
                #         job.ps_mem = 1000

                if job.ps_gpu and job.num_ps:
                    gpu += job.ps_gpu*job.num_ps
                if job.worker_gpu and job.num_worker:
                    gpu += job.worker_gpu * job.num_worker
                if job.ps_cpu and job.num_ps:
                    cpu += job.ps_cpu*job.num_ps
                if job.worker_cpu and job.num_worker:
                    cpu += job.worker_cpu*job.num_worker
                if job.ps_mem and job.num_ps:
                    mem += job.ps_mem*job.num_ps
                if job.worker_mem and job.num_worker:
                    mem += job.worker_mem*job.num_worker
        cpus.append(cpu/100)
        gpus.append(gpu/100)
        mems.append(mem/1000)
    print gpus, cpus, mems
    num_server = 500
    server_cpu = 64
    server_gpu = 2

    max_gpu = num_server * server_gpu
    max_cpu = num_server * server_cpu


    # max_gpu = float(max(gpus))
    # max_cpu = float(max(cpus))
    # max_mem = float(max(mems))
    gpus = [100*gpu/max_gpu for gpu in gpus]
    cpus = [100*cpu/max_cpu for cpu in cpus]
    #mems = [100*mem/max_mem for mem in mems]
    x = [i*0.01 for i in range(2400)]
    plt.style.use(["seaborn-bright", "double-figure.mplstyle"])
    fig, ax = plt.subplots()
    ax.plot(x, gpus, 'b-')
    # ax.plot(x, cpus, 'g--', label="CPU")
    ax.set_xticks([6*i for i in range(5)])
    # ax.set_xticks([i*0.01 for i in range(1200,2400)] + [i*0.01 for i in range(1200)])
    #plt.plot(x, mems, 'r--', label="MEM")
    plt.ylabel("GPU Usage(%)")
    plt.xlabel("Time(h)")
    # plt.legend(loc='best', shadow=False)
    plt.gcf().tight_layout()
    plt.show()
    fig.savefig("resource_usage_variation.pdf")





# job_arrival_day()
# job_length()
# get_number_of_models()
get_gpu_request()