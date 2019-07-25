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
import layer_optimizer
import layer_exhaustive_searcher
import deconv_exhaustive_searcher

method = None
enable = {
    "combine" : False,
    "split" : False,
}

def setup(meta_data, hardware_constraints):
    global enable, method
    # define the search method
    method = meta_data["method"]

    if meta_data["schedule"]["static"]:
        raise Exception("The static scheduling is not supported"
            " in constrained optimizer.")

    enable["combine"] = meta_data["schedule"]["combine"]
    enable["split"] = meta_data["schedule"]["split"]

def single_layer_optimization(data, sys_info):
    global method
    if method == "Constrained":
        return layer_optimizer.LayerOptimizer(data, sys_info).optimize()
    elif method == "Exhaustive":
        return layer_exhaustive_searcher.LayerExhaustiveSearcher(data, sys_info).optimize()
    else:
        raise Exception("Unknown search method: {}".format(method))

def single_combine_optimization(data, sys_info):
    global method
    if method == "Constrained":
      return layer_optimizer.LayerOptimizer(data, sys_info).optimize()
    elif method == "Exhaustive":
        return deconv_exhaustive_searcher.DeconvExhaustiveSearcher(data, sys_info).optimize()
    else:
        raise Exception("Unknown search method: {}".format(method))


def opti_deconv(layer, sys_info):
    global method, enable
    # collect individual result from sub_kernels
    subs = []

    # if the convolution size is odd;
    if layer["kernel"][0]%2 == 1:
        add_one = [(i+1)/2 for i in layer["kernel"]]
        sub_one = [i/2 for i in layer["kernel"]]
        sub1 = dict(layer)
        sub1["kernel"] = [add_one[0], add_one[1]]
        sub2 = dict(layer)
        sub2["kernel"] = [add_one[0], sub_one[1]]
        sub3 = dict(layer)
        sub3["kernel"] = [sub_one[0], add_one[1]]
        sub4 = dict(layer)
        sub4["kernel"] = [sub_one[0], sub_one[1]]

        if enable["combine"]:
            subs.append(single_combine_optimization(layer, sys_info))
        else:
            res1 = single_layer_optimization(sub1, sys_info)
            subs.append(res1)
            res2 = single_layer_optimization(sub2, sys_info)
            subs.append(res2)
            res3 = single_layer_optimization(sub3, sys_info)
            subs.append(res3)
            res4 = single_layer_optimization(sub4, sys_info)
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
            subs.append(single_layer_optimization(sub4, sys_info))
        else:
            # without combining sub-kernels 
            res = single_layer_optimization(sub, sys_info)
            # times 4 of each individual sub-kernel"s
            # memory traffic and cycles.
            res["total_traffic"] = res["total_traffic"]*4
            res["total_cycle"] = res["total_cycle"]*4
            subs.append(res)

    return subs

# the main routine of optimizing the dnn.
def opti_dnn(meta_data, hardware_constraints):
    # set up the configurations;
    setup(meta_data, hardware_constraints)
    dnn = meta_data["dnn"]
    sys_info = meta_data["system_info"]

    results = []

    # optimize for each layer
    for i in range(len(dnn)):
        layer = dnn[i]
        # start to optimize ordinary Conv layer.
        data = dict(layer)

        # check if this layer is Deconv, True == YES
        if layer["Deconv?"] == True:
            if enable["split"]:
                # if split the deconv into smaller ones
                results.append({
                        "data" : data,
                        "result" : opti_deconv(layer, sys_info)
                        })
            else:
                # scale up the ifmap to the ifmap based on the stride size.
                data["ifmap"][0] = layer["ifmap"][0]*2/layer["stride"]
                data["ifmap"][1] = layer["ifmap"][1]*2/layer["stride"]
                results.append({
                        "data" : data,
                        "result" : single_layer_optimization(data, sys_info)
                        })
        else:
            data["ofmap"] = [0,0]
            # scale down the ifmap to the ifmap based on the stride size.
            data["ofmap"][0] = layer["ifmap"][0]/layer["stride"]
            data["ofmap"][1] = layer["ifmap"][1]/layer["stride"]

            results.append({
                        "data" : data,
                        "result" : single_layer_optimization(data, sys_info)
                        })

        # append last result into meta_data
        meta_data["dnn"][i]["result"] = results[-1]

    return results
