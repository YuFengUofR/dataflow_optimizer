#!/usr/bin/python2.7
# own module
from memory_controller import *
from systolic_array import *
from onchip_buffer import *

# public library
import cv2 as cv
import math
import matplotlib
import matplotlib.pyplot as plt

# info for systolic array
sys_arr_size = 16

# info for weights
w_size = 4
# number of weight load together to the l2 cache
w_number = 128

# input layer dimension
height = 96
width = 192
channel = 256

def plotMemoryUsage(keys, total_mems):
    keys_log2 = np.log2(keys)


    plt.rc('font', size=10)
    ax1 = plt.figure(figsize=(6, 3)).add_subplot(111)
    ax1.set_ylabel('Memory Size (in log2)', fontsize=12, fontweight='bold')
    ax1.set_xscale('log', basex=2)
    plt.setp(ax1.spines.values(), linewidth=2)

    ax1.set_xlabel('Tile size (x*y)', fontsize=12, fontweight='bold')
    # p1 = ax1.bar(keys_log2, total_mems, 0.4, align='center',color='#71985E',\
    #     edgecolor=['k']*len(total_mems), linewidth=2, hatch="/");

    p1 = ax1.plot(keys, total_mems, color='#71985E', linestyle='none',\
            linewidth=2, markeredgecolor='k', marker='^', markersize=8);

    plt.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
    plt.subplots_adjust(left=0.10, bottom=0.20, right=0.95, top=0.9,
                wspace=0.2, hspace=0.2)

    ax1.tick_params(axis="y",direction="in")
    ax1.tick_params(axis="x",direction="in")
    # ax1.set_ylim(0, 150)
    plt.grid(color='grey', which='major', axis='y', linestyle='--') 
    # plt.legend((p1[0], p2[0]), ('Batch', 'Traffic'), \
    #         bbox_to_anchor=(0., 1.01, 1., .101), loc=3,
    #         ncol=2, borderaxespad=0., frameon=False)
    ax1.set_axisbelow(True)
    
    plt.savefig("sched_mem.pdf");

def plotMemoryTraffic(keys, w_batchs, total_traffics):
    plt.rc('font', size=10)
    ax1 = plt.figure(figsize=(6, 3)).add_subplot(111)
    ax1.set_ylabel('Batch Size', fontsize=12, fontweight='bold')
    ax1.set_xscale('log', basex=2)
    plt.setp(ax1.spines.values(), linewidth=2)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Memory Traffic in Log10', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Tile size (x*y)', fontsize=12, fontweight='bold')
    # p1 = ax1.bar(keys, total_mem, 0.4, align='center',color='#71985E',\
    #     edgecolor=['k']*len(ebs_axis_ls), linewidth=2, hatch="/");
    
    p1 = ax1.plot(keys, w_batchs, color='#FFBF56', linestyle='--',\
            linewidth=2, markeredgecolor='k', marker='^', markersize=8);
    p2 = ax2.plot(keys, total_traffics, color='#8154D1', linestyle=':',\
            linewidth=2, markeredgecolor='k', marker='o', markersize=8);

    plt.subplots_adjust(left=0.1, bottom=0.20, right=0.9, top=0.9,
                wspace=0.2, hspace=0.2)

    ax1.tick_params(axis="y",direction="in")
    ax2.tick_params(axis="y",direction="in")
    ax1.tick_params(axis="x",direction="in")
    ax1.set_ylim(0, 150)
    ax2.set_ylim(7.0, 10.0)
    plt.grid(color='grey', which='major', axis='y', linestyle='--') 
    plt.legend((p1[0], p2[0]), ('Batch', 'Traffic'), \
            bbox_to_anchor=(0., 1.01, 1., .101), loc=3,
            ncol=2, borderaxespad=0., frameon=False)
    ax1.set_axisbelow(True)
    
    plt.savefig("sched_traffic.pdf");
    

def optimizeLayer(height, width, channel, w_number):
    
    # buffer size
    buffer_size = 1*1024*1024
    # initialize components for simulation
    # mc = MemoryController(sys_arr_size, tile_x, tile_y, w_size)
    # # sys_arr = SystolicArray(sys_arr_size, mc)
    # onchip_buffer = OnchipBuffer(buffer_size, sys_arr_size, tile_x, tile_y, w_size, block_x)
    
    # read the data of a layer frist 
    # mc.readInput(width, height, channel)
    # mc.readWeight(width, height, channel, w_number)


    memory_traffic = {}


    # simulation loops
    for block_y in range(1, height/2+1):
        if (height % block_y != 0):
            continue
        tile_y = height/block_y

        for block_x in range(1, width/2+1):
            # print(block_x, block_sy)
            if (width % block_x != 0):
                continue
            tile_x = width/block_x  

            for w_batch in range(1, w_number+1):
                in_mem_size = (tile_x*tile_y+w_batch*w_size*w_size)*channel
                out_men_size = (tile_x*tile_y*2*2)*w_batch

                if (out_men_size+in_mem_size < 0.9*buffer_size):
                    total_mem = in_mem_size + out_men_size
                    # calculate number of round of loading weights
                    w_round = math.ceil(float(w_number)/w_batch)
                    if (block_y == 1 and block_x == 1):
                        total_traffic = (tile_x*tile_y)*channel \
                                        + (tile_y*tile_x*2*2)*w_number \
                                        + w_size*w_size*channel*w_number
                    else:
                        total_traffic = total_mem*block_x*block_y*w_round

                    memory_traffic[(tile_x, tile_y)] = \
                            (w_batch, w_round, in_mem_size, out_men_size, total_mem, total_traffic)
                else:
                    break


    result = {}
    for (x, y), (w_batch, w_round, in_mem_size, out_men_size, total_mem, total_traffic) in memory_traffic.iteritems():
        result[x*y] = (w_batch, total_mem, total_traffic)

    keys = result.keys()
    keys.sort()
    print(keys)
    w_batchs = []
    total_mems = []
    total_traffics = []

    for k in keys:
        v = result[k]
        w_batchs.append(v[0])
        total_mems.append(v[1])
        total_traffics.append(np.log10(v[2]))

    plotMemoryTraffic(keys, w_batchs, total_traffics)
    plotMemoryUsage(keys, total_mems)

    


if __name__== '__main__':
    optimizeLayer(height, width, channel, w_number)


