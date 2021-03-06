import os
import csv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
import matplotlib.transforms
import matplotlib
import time
import datetime
import numpy as np
import ast
import yaml

from matplotlib.ticker import MaxNLocator


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

    def print_resr_length(self):
        print "model:", self.model, "duration:", self.duration, "num_ps:", self.num_ps, "num_worker:", self.num_worker, \
            "ps_cpu:", self.ps_cpu, "worker_cpu:", self.worker_gpu, "worker_gpu:", self.worker_gpu


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
    plt.style.use(["seaborn-bright", "double-figure.mplstyle"])
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
    bins = [_*3600/3 for _ in range(0,24*3*7)]
    counts, bin_edges = np.histogram(relative_starts, bins=bins)
    dt_list = [datetime.datetime.fromtimestamp(_+ts) for _ in bins]
    dates = matplotlib.dates.date2num(dt_list)
    # print "dates", dates
    # print "counts", counts
    fig, ax = plt.subplots()
    ax.plot_date(dates[1:], counts, 'b-')
    # from matplotlib.dates import MO, TU, WE, TH, FR, SA, SU
    # loc = matplotlib.dates.WeekdayLocator(byweekday=(MO, TU, WE, TH, FR, SA, SU))
    # print "loc", loc
    # ax.xaxis.set_major_locator(loc)
    index_ls = ['MO', 'TU', 'WE', 'TH', 'FR', 'SA', 'SU']
    # plt.xticks(loc, index_ls)
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%A'))
    # ax.tick_params(direction='out', pad=50)
    ax.set_xticklabels(index_ls)
    ax.yaxis.set_major_locator(mtick.MaxNLocator(5))
    # ax.xaxis.set_major_locator(mtick.MaxNLocator(8))
    plt.ylabel("Job arrival rate")
    # plt.setp(ax.xaxis.get_majorticklabels(), ha="left")
    # Create offset transform by 5 points in x direction
    dx = 25 / 72.;
    dy = 0 / 72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)

    # apply offset transform to all x ticklabels.
    for label in ax.xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    # plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.show()
    fig.savefig("job_arrival_rate.pdf")



def job_length():
    plt.style.use(["seaborn-bright", "double-figure.mplstyle"])
    durations = []
    for job in jobs:
        if job.dist:
            durations.append(job.duration/60)
    print "average job duration: ", sum(durations)/len(durations), "min"
    # plt.hist(durations, bins=20, cumulative=True, label='CDF', normed=True, histtype='step')
    bins = [_ for _ in range(0, max(durations), max(durations)/1000)]
    counts, bin_edges = np.histogram(durations, bins=bins)
    cdf = np.cumsum(counts, dtype=float)
    cdf /= cdf[-1]
    cdf = np.append(np.array([0.0]), cdf)
    fig, ax = plt.subplots()
    ax.plot(bin_edges, cdf*100, 'b-')
    ax.xaxis.set_major_locator(MaxNLocator(40))
    plt.xlabel("Duration (min)")
    plt.ylabel("CDF (%)")
    plt.ylim(0,100)
    plt.xlim(left=1)

    plt.xscale('log')
    plt.gcf().subplots_adjust(bottom=0.2, left=0.18)
    plt.tight_layout()
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
    print "# of models trained for more than five", sum([m > 5 for m in models.values()])
    print "# of models trained for more than ten", sum([m >= 10 for m in models.values()])
    # plt.hist(models.values(), bins=20)
    # plt.show()

    more_models = dict()
    for k, v in models.items():
        if v > 1:
            more_models[k] = v

    plt.hist(more_models.values(), bins=50)
    plt.show()


def fit_resource_speed_curve():
    models = dict()
    cand_models = set()
    for job in jobs:
        if job.dist:
            if job.model not in models.keys():
                models[job.model] = 1
            else:
                models[job.model] += 1
            if models[job.model] >= 10 and job.model not in cand_models:
                cand_models.add(job.model)
    print cand_models
    cand_jobs = dict()
    for job in jobs:
        if job.dist and job.model in cand_models:
            if job.model not in cand_jobs.keys():
                cand_jobs[job.model] = []
            cand_jobs[job.model].append(job)

    for k, v in cand_jobs.items():
        # print  k, v
        for job in v:
            job.print_resr_length()
        print '--------------------------'


def est_interference():
    plt.style.use(["seaborn-bright", "double-figure.mplstyle"])
    models = dict()
    cand_models = set()
    for job in jobs:
        if job.dist:
            if job.model not in models.keys():
                models[job.model] = 1
            else:
                models[job.model] += 1
            if models[job.model] >= 2 and job.model not in cand_models:
                cand_models.add(job.model)
    #print cand_models
    cand_jobs = dict()
    for job in jobs:
        if job.dist and job.model in cand_models:
            if job.model not in cand_jobs.keys():
                cand_jobs[job.model] = []
            cand_jobs[job.model].append(job.duration)

    errors = []
    for k, v in cand_jobs.items():
        error = np.abs(np.std(np.array(v))/np.average(np.array(v)))
        errors.append(error)
    print 'length: ', len(errors)
    print np.average(errors)
    print max(errors)
    print sum([error>=1 for error in errors])/float(len(errors))

    bins = 1000
    counts, bin_edges = np.histogram(errors, bins=bins)
    cdf = np.cumsum(counts, dtype=float)
    cdf /= cdf[-1]
    cdf = np.append(np.array([0.0]), cdf)
    fig, ax = plt.subplots()
    ax.plot(bin_edges*100, cdf*100, 'b-')
    # ax.xaxis.set_major_locator(MaxNLocator(5))
    plt.xlabel("std/avg (%)")
    plt.ylabel("CDF (%)")
    plt.ylim(0,100)
    # plt.xlim(left=0)

    plt.xscale('log')
    plt.gcf().subplots_adjust(bottom=0.2, left=0.18)
    plt.tight_layout()
    plt.show()
    fig.savefig("job_interference_cdf.pdf")


def get_gpu_request():
    candidates = []
    for job in jobs:
        start = Job.format_time(job.start)
        if start.tm_year == 2018 and start.tm_mon == 9 and start.tm_mday >= 1 and start.tm_mday <= 5:
            candidates.append(job)
    # calculate Sep. 5
    dt_obj = datetime.datetime(2018, 9, 3, 0, 0, 0) # start from 12:00 noon, so need to adjust xticks
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
    max_gpu = max(gpus)
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
    plt.ylabel("GPU Utilization(%)")
    plt.xlabel("Time (hours)")
    # plt.legend(loc='best', shadow=False)
    plt.gcf().tight_layout()
    plt.show()
    fig.savefig("resource_usage_variation.pdf")



def job_arrival_pattern():
    starts = []
    for job in jobs:
        start = Job.format_time(job.start)
        if start.tm_year == 2018 and start.tm_mon == 8 and start.tm_mday >= 20 and start.tm_mday <=22:
            starts.append(job.start)
    dt_obj = datetime.datetime(2018, 8, 20, 0, 0, 0)
    ts = time.mktime(dt_obj.timetuple())
    relative_starts = [_ - ts for _ in starts]
    arrival = [0 for _ in range(24*3)]
    for start in relative_starts:
        hour = int(start/3600)
        arrival[hour] += 1
    print arrival


def job_length_pattern():
    max_length = 1000
    durations = dict()

    for job in jobs:
        if job.dist:
            duration = job.duration/60
            index = min([7, duration/(max_length/8)])
            if index not in durations.keys():
                durations[index] =[]
            durations[index].append(duration)

    tot_num = 0
    job_length_dist = [0 for _ in range(8)]
    for k, v in durations.items():
        job_length_dist[k] = sum(v)/len(v)
        tot_num += len(v)
    job_length_prob = [len(v)/float(tot_num) for k, v in durations.items()]
    print job_length_prob, job_length_dist

    sum_length = 0
    for i in range(8):
        sum_length += (job_length_dist[i] * job_length_prob[i])
    print sum_length




# job_arrival_day()
# job_length()
# get_number_of_models()
# get_gpu_request()
# fit_resource_speed_curve()
# job_arrival_pattern()
# job_length_pattern()
est_interference()