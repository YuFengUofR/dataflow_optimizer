#!/usr/bin/python2.7
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import FuncFormatter
import numpy as np

# if it is static schedule the buffer 
static_schedule = False

# add suffix to every plot for one configuration profiling
suffix = "combine"

# import my own modules
from dnn_analysis import *

if static_schedule:
    from layer_static import optimize, setup_hardware
else:
    from layer_optimizer import optimize, setup_hardware

# if profile in details for one particular network.
detail_profile = True

# if we combine two different sub-kernels and optimize them 
# together, then, enable this switch
enable_combine = True

# a list to store the dnn configuration 
dnn = []

# a list to store all the optimization results
results = []

# import dnn network descrtiption into the system;
# the format for one DNN layer is: 
# (width, height, in_channel, out_channel,
#  kenrel_width, kernel_height, stride, Deconv?)
def import_dnn(filename=None):
    # clear all the previous contents;
    del dnn[:]
    ifmap_dim = [512, 384, 6]
    weight_dim = []

    # The weight input format as follows: 
    # [out_channel,kenrel_width,kernel_height,stride,Deconv?]
    for line in open(filename):
        ls = line.strip().split(",")
        weight_dim.append([int(ls[0]), int(ls[1]), int(ls[2]),\
                             int(ls[3]), ls[4] == 'True'])

    for w in weight_dim:
        # first append the necessary information to compute this Conv layer 
        dnn.append(list(ifmap_dim+w))
        # if it is Deconv;
        if w[-1]:
            # increase the deconv ofmap by two;
            # (considered the Deconv always scale the ifmap by 2)
            ifmap_dim = [ifmap_dim[0]*2, ifmap_dim[1]*2, w[0]]
        else: # if it is Conv
            # scale down the ifmap dimemsion by stride;
            ifmap_dim = [ifmap_dim[0]/w[-2], ifmap_dim[1]/w[-2], w[0]]


# The hardware constraints are:
#   1. the on-chip buffer size; 
#   2. the memory bandwidth; (Unit in bytes/cycle) 
#   3. the systolic array size;
def hardware_constraints(sa_size=16.0, mem_bw=16.0, buf=1048576.0):
    systolic_arr_size = sa_size;
    memory_bandwidth = mem_bw;
    buffer_size = buf;
    return [systolic_arr_size, memory_bandwidth, buffer_size]

# the main routine of optimizing the dnn.
def opti_dnn():
    global results
    # clear the result first
    del results[:]

    # optimize for each layer
    for layer in dnn:
        print("[Layer]",layer)

        # check if this layer is Deconv, True == YES
        if layer[-1] == True:
            # if the convolution size is odd;
            if layer[5]%2 == 1:
                sub1 = list(layer)
                sub1[4] = (sub1[4]+1)/2
                sub1[5] = (sub1[5]+1)/2
                # set sub_res1 == None
                sub_res1 = None
                if enable_combine:
                    sub1[3] = sub1[3]*2
                    results.append(optimize(sub1))
                else:
                    res1 = optimize(sub1)
                    res1[0] = res1[0]*2
                    res1[1] = res1[1]*2
                    results.append(res1)

                # handle the second sub-kernel    
                sub2 = list(layer)
                sub2[4] = (sub2[4]-1)/2
                sub2[5] = (sub2[5]-1)/2
                if enable_combine:
                    sub2[3] = sub2[3]*2
                    results.append(optimize(sub2))
                else:
                    res2 = optimize(sub2)
                    res2[0] = res2[0]*2
                    res2[1] = res2[1]*2
                    results.append(res2)
                # based on whether combine or not,
                # compute the total cycles and averaged utilization
                # (width, height, in_channel, out_channel,
                #  kenrel_width, kernel_height, stride, Deconv?) 
                


            # if the convolution size is even;
            else:
                sub = list(layer)
                sub[4] = sub[4]/2
                sub[5] = sub[5]/2
                if enable_combine:
                    # this will consider four same-size sub-kernels 
                    # as one sub-kernel with more channels
                    sub[3] = sub[3]*4
                    results.append(optimize(sub))
                else:
                    # without combining sub-kernels 
                    res = optimize(sub)
                    # times 4 of each individual sub-kernel's
                    # memory traffic and cycles.
                    res[0] = res[0]*4
                    res[1] = res[1]*4
                    results.append(res)
        else:
            # start to optimize ordinary Conv layer.
            sub = list(layer)
            # scale down the ifmap to the ifmap based on the stride size.
            sub[0] = layer[0]/layer[-2]
            sub[1] = layer[1]/layer[-2]
            results.append(optimize(sub))

    for res in results:
        print(res)

    return results


if __name__== '__main__':
    # import the dnn
    import_dnn("dnns/flowNetS.txt")

    # check which characterization you want to proceed
    if detail_profile:
        # set up the hardware configuration
        setup_hardware(hardware_constraints())
        # start the optimization main routine
        res = opti_dnn()
        # plot the result of each layer
        plot_util_dnn(res, suffix)
        profile_layer_cycle(res, suffix)
    else:
        # profile systolic array size impacts
        profile_sa_size(8, 36, 4)

        # profile bandwidth impacts
        profile_bw_size(-3, 8)

        # profile buffer size impacts
        profile_buf_size(1, 10)




