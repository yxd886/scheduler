import os

'''
analyze trace in arnold
'''

def read_linearity_speed():
    dnns = ['resnet_50', 'resnet_152', 'vgg_16', 'inception-bn', 'alexnet']
    gpu_models = ['GeForce-GTX-1080-Ti']
    num_ps_workers= [(i,i) for i in range(1,9,1)]

    speed_maps = dict()

    for gpu_model in gpu_models:
        for dnn in dnns:
            dnn_speeds = []
            task_name = 'task-' + dnn + '-' + gpu_model
            for num_ps, num_workers in num_ps_workers:
                trial_name = task_name + "/" + "ps-" + str(num_ps) + "-worker-" + str(num_workers)
                log_file = trial_name + "/worker-1/results/training.log"
                speeds = []
                with open(log_file, 'r') as f:
                    for line in f:
                        if "Epoch" in line and "Batch" in line and "Speed" in line:
                            start = line.index('Speed') + len("Speed") + 2
                            end = line.index('samples/sec') -1
                            speed = float(line[start:end])
                            speeds.append(speed)
                assert len(speeds)
                avg_speed = float('%.2f'%(sum(speeds)/len(speeds)))
                dnn_speeds.append(avg_speed)
            speed_maps[dnn] = dnn_speeds
    print speed_maps

linearity_speed_maps = {'resnet_152': [33.18, 18.09, 18.64, 28.47, 17.29, 27.16, 17.23, 16.46], 'inception-bn': [101.42, 86.25, 77.82, 75.82, 66.73, 40.52, 52.95, 49.29], 'vgg_16': [11.53, 13.58, 15.63, 13.38, 13.79, 12.9, 10.74, 13.19], 'resnet_50': [43.28, 43.75, 40.92, 61.1, 40.97, 60.22, 39.05, 45.92], 'alexnet': [31.72, 33.98, 45.1, 47.83, 21.96, 43.5, 44.16, 12.99]}

# read_speed()

def read_ratio_speed():
    dnns = ['resnet_50', 'resnet_152', 'vgg_16', 'inception-bn', 'alexnet']
    gpu_models = ['GeForce-GTX-1080-Ti']
    num_ps_workers= [(3,9),(4,8),(6,6), (8,4), (9,3)]

    speed_maps = dict()

    for gpu_model in gpu_models:
        for dnn in dnns:
            dnn_speeds = []
            task_name = 'ratio-' + dnn + '-' + gpu_model
            for num_ps, num_workers in num_ps_workers:
                trial_name = task_name + "/" + "ps-" + str(num_ps) + "-worker-" + str(num_workers)
                log_file = trial_name + "/worker-1/results/training.log"
                speeds = []
                with open(log_file, 'r') as f:
                    for line in f:
                        if "Epoch" in line and "Batch" in line and "Speed" in line:
                            start = line.index('Speed') + len("Speed") + 2
                            end = line.index('samples/sec') -1
                            speed = float(line[start:end])
                            speeds.append(speed)
                assert len(speeds)
                avg_speed = float('%.2f'%(sum(speeds)/len(speeds)))
                dnn_speeds.append(avg_speed)
            speed_maps[dnn] = dnn_speeds
    print speed_maps

ratio_speed_maps = {'resnet_152': [14.79, 11.08, 21.8, 17.03, 25.56], 'inception-bn': [29.86, 51.12, 47.43, 58.38, 73.25], 'vgg_16': [7.78, 7.05, 7.35, 8.74, 10.65], 'resnet_50': [29.14, 38.5, 26.29, 44.3, 48.86], 'alexnet': [10.21, 24.26, 16.1, 46.22, 59.86]}

# read_ratio_speed()

def draw_linear(speed_maps):
    import numpy as np
    import matplotlib.pyplot as plt
    plt.style.use(["ggplot", "double-figure.mplstyle"])

    dnns = ['resnet_50', 'resnet_152', 'vgg_16', 'inception-bn', 'alexnet']

    fig, ax = plt.subplots()
    plt.xlabel('# of workers')
    plt.ylabel('Norm. Speed')

    styles = ["b-", "r*-", "c^-", "y--", "kD-"]
    for i in range(len(dnns)):
        dnn = dnns[i]
        x = np.array([_ for _ in range(len(speed_maps[dnn]))])
        y = np.array(speed_maps[dnn])/max(speed_maps[dnn])
        plt.plot(x, y, styles[i%len(styles)], label=dnn)
    legend = ax.legend(loc='best', shadow=False)
    frame = legend.get_frame()
    frame.set_facecolor('1')
    plt.title("PS:Worker=1:1")
    plt.savefig('arnold_linearity.pdf', format='pdf', dpi=1000)
    plt.show()

def draw_ratio(speed_maps):
    # each  model is tested under config: (3,9),(4,8),(6,6), (8,4), (9,3)
    # draw a figure showing the speed under different configs for different models
    # this figure spans one column instead of a half
    # vgg, resnet-152,
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    colors = ['g', 'r', 'b', 'c', 'k']
    i = 0
    for model, speeds in speed_maps.items():
        x = [_ for _ in range(5)]
        ax.plot(x, speeds, colors[i]+'-', label=model)
        i += 1
    legend = ax.legend(loc='best', shadow=False)
    plt.show()

    pass

# draw_linear(linearity_speed_maps)
draw_ratio(ratio_speed_maps)