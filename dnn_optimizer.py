#!/usr/bin/python2.7
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import FuncFormatter
import numpy as np
import scipy
import sys


# import my own modules
from dnn_analysis import *
import layer_static, layer_optimizer

enable = {
    "combine" : False,
    "split" : False,
}

# a list to store all the optimization results
results = []

def setup(meta_data, hardware_constraints):
    global enable

    if meta_data["schedule"]["static"]:
        layer_static.setup_hardware(hardware_constraints)
    else:
        layer_optimizer.setup_hardware(hardware_constraints)

    enable["combine"] = meta_data["schedule"]["combine"]
    enable["split"] = meta_data["schedule"]["split"]


def opti_deconv(layer):
    # collect individual result from sub_kernels
    subs = []

    # if the convolution size is odd;
    if layer["kernel"][0]%2 == 1:
        sub1 = dict(layer)
        sub1["kernel"][0] = (sub1["kernel"][0]+1)/2
        sub1["kernel"][1] = (sub1["kernel"][1]+1)/2
        sub2 = dict(layer)
        sub2["kernel"][0] = (sub2["kernel"][0]+1)/2
        sub2["kernel"][1] = (sub2["kernel"][1]-1)/2
        sub3 = dict(layer)
        sub3["kernel"][0] = (sub3["kernel"][0]-1)/2
        sub3["kernel"][1] = (sub3["kernel"][1]+1)/2
        sub4 = dict(layer)
        sub4["kernel"][0] = (sub4["kernel"][0]-1)/2
        sub4["kernel"][1] = (sub4["kernel"][1]-1)/2
        
        if enable_combine:
            subs.append(layer_optimizer.optimize(layer))
        else:
            res1 = layer_optimizer.optimize(sub1)
            subs.append(res1)
            res2 = layer_optimizer.optimize(sub2)
            subs.append(res2)
            res3 = layer_optimizer.optimize(sub3)
            subs.append(res3)
            res4 = layer_optimizer.optimize(sub4)
            subs.append(res4)

    # if the convolution size is even;
    else:
        sub = dict(layer)
        sub["kernel"][0] = sub["kernel"][0]/2
        sub["kernel"][1] = sub["kernel"][1]/2
        if enable_combine:
            # this will consider four same-size sub-kernels 
            # as one sub-kernel with more channels
            sub["out_channel"] = sub["out_channel"]*4
            subs.append(layer_optimizer.optimize(sub))
        else:
            # without combining sub-kernels 
            res = layer_optimizer.optimize(sub)
            # times 4 of each individual sub-kernel's
            # memory traffic and cycles.
            res["total_traffic"] = res["total_traffic"]*4
            res["total_cycle"] = res["total_cycle"]*4
            subs.append(res)

    ret = [0, 0, 0, 0]
    for item in subs:
        ret = [x+y for x,y in zip(ret,item)]

    # this is used to divide the length of the subs-kernel
    # to get the utilization of SA ans buf.
    ret[2] /= len(subs)
    ret[3] /= len(subs)
    results.append(ret)
    # sum all the results
    return ret

# the main routine of optimizing the dnn.
def opti_dnn(meta_data, hardware_constraints):
    # set up the configurations;
    setup(meta_data, hardware_constraints)
    dnn = meta_data["dnn"]

    results = []

    # optimize for each layer
    for i in range(len(dnn)):
        layer = dnn[i]

        # check if this layer is Deconv, True == YES
        if layer["Deconv?"] == True:
            if enable["split"]:
                # if split the deconv into smaller ones
                results.append({
                        "data" : data,
                        "result" :opti_deconv(layer)
                        })
            else:
                # start to optimize ordinary Conv layer.
                data = dict(layer)
                # scale up the ifmap to the ifmap based on the stride size.
                data["ifmap"][0] = layer["ifmap"][0]*2/layer["stride"]
                data["ifmap"][1] = layer["ifmap"][1]*2/layer["stride"]
                results.append({
                        "data" : data,
                        "result" :layer_optimizer.LayerOptimizer(data).optimize()
                        })
        else:
            # start to optimize ordinary Conv layer.
            data = dict(layer)
            data["ofmap"] = [0,0]
            # scale down the ifmap to the ifmap based on the stride size.
            data["ofmap"][0] = layer["ifmap"][0]/layer["stride"]
            data["ofmap"][1] = layer["ifmap"][1]/layer["stride"]

            results.append({
                        "data" : data,
                        "result" :layer_optimizer.LayerOptimizer(data).optimize()
                        })

        # append last result into meta_data
        meta_data["dnn"][i]["result"] = results[-1]

    return results
