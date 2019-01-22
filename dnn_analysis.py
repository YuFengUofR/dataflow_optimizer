#!/usr/bin/python2.7
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import FuncFormatter
import numpy as np

# This module is used for visualizing the result.


# This is a function to plot the utilization of buffer 
# and systolic array for a single DNN under one particular
# hardware configuration
def plot_util_dnn(res, suffix):
    # initialize the array from the result first.
    sa_util = []
    buf_util = []
    comp_bound = {'x':[], 'y': []}
    mem_bound = {'x':[], 'y': []}
    cnt = 1

    # the content in the result is listed as:
    for item in res:
        sa_util.append(item[2])
        buf_util.append(item[3])
        if item[-1] == 'C':
            comp_bound['x'].append(cnt)
            comp_bound['y'].append(item[2]*item[3])
        else:
            mem_bound['x'].append(cnt)
            mem_bound['y'].append(item[2]*item[3])
        cnt += 1

    x_axis_ls = range(1, len(res)+1)
    plt.rc('font', size=10)
    ax1 = plt.figure(figsize=(8, 3)).add_subplot(111)
    ax1.set_ylabel('PE util', fontsize=14, fontweight='bold')
    plt.xticks(rotation=60)
    plt.setp(ax1.spines.values(), linewidth=2)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Buf Util', fontsize=14, fontweight='bold')

    p1 = ax1.plot(x_axis_ls, sa_util, color='#71985E', linestyle=':', linewidth=2, \
                marker='d',markersize=8, markeredgewidth=1.5, markeredgecolor='k');
    
    p2 = ax2.plot(x_axis_ls, buf_util, color='#FFBF56', linestyle=':', linewidth=2, \
                    marker='o',markersize=8, markeredgewidth=1.5, markeredgecolor='k');

    p3 = ax1.bar(comp_bound['x'], comp_bound['y'], 0.5, align='center',color='#8154D1',\
        edgecolor=['k']*len(x_axis_ls), linewidth=1.5, hatch="/");
    
    p4 = ax2.bar(mem_bound['x'], mem_bound['y'], 0.5, align='center', \
        color='#5b87f2', edgecolor=['k']*len(x_axis_ls), linewidth=1.5, hatch="\\");
    plt.subplots_adjust(left=0.1, bottom=0.25, right=0.9, top=0.9,
                wspace=0.2, hspace=0.2)
    plt.xticks(x_axis_ls, [ "Layer" + str(n) for n in x_axis_ls])
    ax2.set_ylim(0.0, 1.0)
    ax1.set_ylim(0.0, 1.0)
    ax1.tick_params(axis="y",direction="in")
    ax2.tick_params(axis="y",direction="in")
    ax1.tick_params(axis="x",direction="in")
    plt.grid(color='grey', which='major', axis='y', linestyle='--')
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
    plt.legend((p1[0], p2[0], p3[0], p4[0]), ('SA util', 'Buf util',\
                'Comp-bound', 'Mem-bound'), bbox_to_anchor=(0., 1.01, 1., .101), loc=3,
           ncol=4, mode="expand", borderaxespad=0., frameon=False)
    ax1.set_axisbelow(True)
    
    plt.savefig(suffix+"_layer_util.pdf");


def profile_layer_cycle(res, suffix):
    # initialize the array from the result first.
    total_transfer = []
    total_cycle = []

    # the content in the result is listed as:
    for item in res:
        total_transfer.append(np.log(item[0]))
        total_cycle.append(np.log(item[1]))

    x_axis_ls = range(1, len(res)+1)
    plt.rc('font', size=10)
    ax1 = plt.figure(figsize=(8, 3)).add_subplot(111)
    ax1.set_ylabel('Memory traffic', fontsize=14, fontweight='bold')
    plt.xticks(rotation=60)
    plt.setp(ax1.spines.values(), linewidth=2)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Cycle in Logscale', fontsize=14, fontweight='bold')

    p1 = ax1.plot(x_axis_ls, total_transfer, color='#71985E', linestyle=':', linewidth=2, \
                marker='d',markersize=8, markeredgewidth=1.5, markeredgecolor='k');
    
    p2 = ax2.plot(x_axis_ls, total_cycle, color='#FFBF56', linestyle=':', linewidth=2, \
                    marker='o',markersize=8, markeredgewidth=1.5, markeredgecolor='k');
    plt.subplots_adjust(left=0.1, bottom=0.25, right=0.9, top=0.9,
                wspace=0.2, hspace=0.2)
    ax1.set_ylim(13.0, 17.0)
    ax2.set_ylim(11.0, 17.0)
    plt.xticks(x_axis_ls, [ "Layer" + str(n) for n in x_axis_ls])
    ax1.tick_params(axis="y",direction="in")
    ax2.tick_params(axis="y",direction="in")
    ax1.tick_params(axis="x",direction="in")
    plt.grid(color='grey', which='major', axis='y', linestyle='--')
    plt.legend((p1[0], p2[0]), ('mem. traffic', 'cycle'), \
            bbox_to_anchor=(0., 1.01, 1., .101), loc=3,
            ncol=4, mode="expand", borderaxespad=0., frameon=False)
    ax1.set_axisbelow(True)
    
    plt.savefig(suffix+"_layer_cycle.pdf");


'''
The functions below are to profile the impacts of different bandwidth,
buffer size, and systolic array size on overall system.
'''
def profile_sa_size(low, high, step):
    arr = []
    # systolic_arr_size, memory_bandwidth, buffer_size
    config = hardware_constraints()
    for size in range(low, high+step, step):
        print("SIZE", size)
        config[0] = float(size)
        print(config)
        setup_hardware(config)
        res = opti_dnn()
        arr.append([item[2:4] for item in res])

    # gather results
    res = {"sa_avg":[], "sa_std":[], "buf_avg":[], "buf_std":[]}

    for ls in arr:
        res["sa_avg"].append(np.mean([i[0] for i in ls]))
        res["sa_std"].append(np.std([i[0] for i in ls]))
        res["buf_avg"].append(np.mean([i[1] for i in ls]))
        res["buf_std"].append(np.mean([i[1] for i in ls]))

    print(res)
    plt.rc('font', size=10)
    ax1 = plt.figure(figsize=(8, 3)).add_subplot(111)
    ax1.set_ylabel('PE Util', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Systolic Array Size', fontsize=14, fontweight='bold')
    plt.setp(ax1.spines.values(), linewidth=2)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Buf Util', fontsize=14, fontweight='bold')
    p1 = ax1.bar([i - 0.3 for i in range(len(res['sa_avg']))], res['sa_avg'], \
        0.3, yerr=res['sa_std'], align='center',color='#8154D1',\
        edgecolor=['k']*len(res['sa_avg']), linewidth=1.5, hatch="/");
    
    p2 = ax2.bar(range(len(res['buf_avg'])), res['buf_avg'], 0.3, yerr=res['buf_std'], align='center', \
        color='#5b87f2', edgecolor=['k']*len(res['buf_avg']), linewidth=1.5, hatch="\\");
    plt.subplots_adjust(left=0.1, bottom=0.20, right=0.9, top=0.9,
                wspace=0.2, hspace=0.2)
    ax2.set_ylim(0.0, 1.0)
    ax1.set_ylim(0.0, 1.0)
    plt.xticks(range(len(res['sa_avg'])), range(low, high+step, step))
    plt.legend((p1[0], p2[0]), ('SA util', 'Buf util'), \
            bbox_to_anchor=(0., 1.01, 1., .101), loc=3, \
            ncol=4, mode="expand", borderaxespad=0., frameon=False)
    ax1.set_axisbelow(True)
    
    plt.savefig("profile_sa.pdf");

def profile_bw_size(low, high):
    arr = []
    # systolic_arr_size, memory_bandwidth, buffer_size
    config = hardware_constraints()
    for i in range(low, high):
        config[1] = float(pow(2, i))
        setup_hardware(config)
        res = opti_dnn()
        arr.append([item[2:4] for item in res])

    # gather results
    res = {"sa_avg":[], "sa_std":[], "buf_avg":[], "buf_std":[]}

    for ls in arr:
        res["sa_avg"].append(np.mean([i[0] for i in ls]))
        res["sa_std"].append(np.std([i[0] for i in ls]))
        res["buf_avg"].append(np.mean([i[1] for i in ls]))
        res["buf_std"].append(np.mean([i[1] for i in ls]))

    print(res)
    plt.rc('font', size=10)
    ax1 = plt.figure(figsize=(8, 3)).add_subplot(111)
    ax1.set_ylabel('PE Util', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Memory Bandwidth (B/cycle)', fontsize=14, fontweight='bold')
    plt.setp(ax1.spines.values(), linewidth=2)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Buf Util', fontsize=14, fontweight='bold')
    p1 = ax1.bar([i - 0.3 for i in range(len(res['sa_avg']))], res['sa_avg'],\
     0.3, yerr=res['sa_std'], align='center',color='#8154D1',\
        edgecolor=['k']*len(res['sa_avg']), linewidth=1.5, hatch="/");
    
    p2 = ax2.bar(range(len(res['buf_avg'])), res['buf_avg'], 0.3, yerr=res['buf_std'], align='center', \
        color='#5b87f2', edgecolor=['k']*len(res['buf_avg']), linewidth=1.5, hatch="\\");
    plt.subplots_adjust(left=0.1, bottom=0.20, right=0.9, top=0.9,
                wspace=0.2, hspace=0.2)
    ax2.set_ylim(0.0, 1.0)
    ax1.set_ylim(0.0, 1.0)
    plt.xticks(range(len(res['sa_avg'])), [pow(2, i) for i in range(low, high)])
    plt.legend((p1[0], p2[0]), ('SA util', 'Buf util'), \
            bbox_to_anchor=(0., 1.01, 1., .101), loc=3, \
            ncol=4, mode="expand", borderaxespad=0., frameon=False)
    ax1.set_axisbelow(True)
    
    plt.savefig("profile_bw.pdf");

def profile_buf_size(low, high):
    arr = []
    # systolic_arr_size, memory_bandwidth, buffer_size
    config = hardware_constraints()
    for size in range(low, high):
        config[2] = config[2]*0.5*size
        print(config)
        setup_hardware(config)
        res = opti_dnn()
        arr.append([item[2:4] for item in res])

    # gather results
    res = {"sa_avg":[], "sa_std":[], "buf_avg":[], "buf_std":[]}

    for ls in arr:
        res["sa_avg"].append(np.mean([i[0] for i in ls]))
        res["sa_std"].append(np.std([i[0] for i in ls]))
        res["buf_avg"].append(np.mean([i[1] for i in ls]))
        res["buf_std"].append(np.mean([i[1] for i in ls]))

    print(res)
    plt.rc('font', size=10)
    ax1 = plt.figure(figsize=(8, 3)).add_subplot(111)
    ax1.set_ylabel('PE Util', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Buffer Size (MB)', fontsize=14, fontweight='bold')
    plt.setp(ax1.spines.values(), linewidth=2)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Buf Util', fontsize=14, fontweight='bold')
    p1 = ax1.bar([i - 0.3 for i in range(len(res['sa_avg']))], res['sa_avg'],\
     0.3, yerr=res['sa_std'], align='center',color='#8154D1',\
        edgecolor=['k']*len(res['sa_avg']), linewidth=1.5, hatch="/");
    
    p2 = ax2.bar(range(len(res['buf_avg'])), res['buf_avg'], 0.3, yerr=res['buf_std'], align='center', \
        color='#5b87f2', edgecolor=['k']*len(res['buf_avg']), linewidth=1.5, hatch="\\");
    plt.subplots_adjust(left=0.1, bottom=0.20, right=0.9, top=0.9,
                wspace=0.2, hspace=0.2)
    ax2.set_ylim(0.0, 1.0)
    ax1.set_ylim(0.0, 1.0)
    plt.xticks(range(len(res['sa_avg'])), [0.5*i for i in range(low, high)])
    plt.legend((p1[0], p2[0]), ('SA util', 'Buf util'), \
            bbox_to_anchor=(0., 1.01, 1., .101), loc=3, \
            ncol=4, mode="expand", borderaxespad=0., frameon=False)
    ax1.set_axisbelow(True)
    
    plt.savefig("profile_buf.pdf");

