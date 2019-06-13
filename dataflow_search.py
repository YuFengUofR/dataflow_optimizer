#!/usr/bin/python2.7
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import FuncFormatter
import argparse
import numpy as np
import scipy
import sys
import pprint

# import my own modules
import dnn_optimizer as opt

# setup the argument parser
argparser = argparse.ArgumentParser('dataflow_search.py')
# input dnn file and output result
argparser.add_argument('--dnnfile', required=True)
argparser.add_argument('--outfile', help="output file to dump all the results")

# other search options
argparser.add_argument('--static', type=bool, default=False, 
                        help="static partition the buffer without dynamically changing")
argparser.add_argument('--split', type=bool, default=False,
                        help="enable to split the convolution kernel into small sub-kernel")
argparser.add_argument('--combine', type=bool, default=False,
                        help="enable to combine the sub-kernels durting compute")
argparser.add_argument('--model_type', default='2D', choices=['2D', '3D'],
                        help='DNN model convolution type: 2D or 3D.')
argparser.add_argument('--ifmap', nargs="+", type=int, required=True, 
                        help="the ifmap dimemsion, order: [W H C] or [W H D C]")
argparser.add_argument('--search_methods', default='Constraint', choices=['Constraint', 'Exhaustive'],
                    help='Dataflow search methods: constraint optoimization or exhaustive search.')

# other hardware configurations
argparser.add_argument('--bufsize', type=float, default=1048576.0*1.5, 
                        help="in Btyes")
argparser.add_argument('--memory_bandwidth', type=float, default=6.4*4,
                        help="in GB/s")
argparser.add_argument('--sa_size', type=float, default=16, 
                        help="Systolic array size")
argparser.add_argument('--bit_width', type=float, default=16, 
                        help="Bit Width of each value (typically, 8-bit, 16-bit, 32-bit)")


args = argparser.parse_args()

# import dnn network descrtiption into the system;
# the format for one DNN layer is: 
# (width, height, in_channel, out_channel,
#  kenrel_width, kernel_height, stride, Deconv?)
def import_dnn(filename, ifmap_dim):
    # a list to store the dnn configuration 
    dnn = []
    weight_dim = []

    # The weight input format as follows: 
    # [out_channel,kenrel_width,kernel_height,stride,Deconv?]
    for line in open(filename):
        ls = line.strip().split(",")
        weight_dim.append([int(ls[0]), int(ls[1]), int(ls[2]),\
                             int(ls[3]), ls[4] == 'True'])

    for w in weight_dim:
        # first append the necessary information to compute this Conv layer 
        dnn.append({"ifmap": ifmap_dim,
                    "out_channel": w[0],
                    "kernel": w[1:3],
                    "stride": w[3],
                    "Deconv?": w[-1]})

        # if it is Deconv;
        if w[-1]:
            # increase the deconv ofmap by two, as default,
            # we only consider stride of 1
            ifmap_dim = [ifmap_dim[0]*2/w[-2], ifmap_dim[1]*2/w[-2], w[0]]
        else: 
            # if it is Conv, scale down the ifmap dimemsion by stride;
            ifmap_dim = [ifmap_dim[0]/w[-2], ifmap_dim[1]/w[-2], w[0]]

    return dnn


# The hardware constraints are:
#   1. the on-chip buffer size; 
#   2. the memory bandwidth; (Unit in bytes/cycle) 
#   3. the systolic array size;
def hardware_constraints(sa_size=24.0, mem_bw=32.0, buf=1048576.0*2.0, bit_width=16.0):
    systolic_arr_size = sa_size;
    memory_bandwidth = mem_bw;
    buffer_size = buf;
    return [systolic_arr_size, memory_bandwidth, buffer_size]

def system_config(args, meta_data):
    # setup the system configuration;
    meta_data["schedule"] = {}
    meta_data["schedule"]["static"] = args.static
    meta_data["schedule"]["split"] = args.split
    meta_data["schedule"]["combine"] = args.combine

    # setup the system;
    meta_data["system_info"] = {}
    meta_data["system_info"]["bufsize"] = args.bufsize
    meta_data["system_info"]["memory_bandwidth"] = args.memory_bandwidth
    meta_data["system_info"]["sa_size"] = args.sa_size
    meta_data["system_info"]["bit_width"] = args.bit_width

    return meta_data

if __name__== '__main__':
    # initialize the result data;
    meta_data = {}

    # setup system configuration;
    meta_data = system_config(args, meta_data)

    # import the dnn
    dnn = import_dnn(args.dnnfile, args.ifmap)
    meta_data["dnn"] = dnn
    hw_constraints = hardware_constraints(sa_size=args.sa_size,
             mem_bw=args.memory_bandwidth, buf=args.bufsize, bit_width=args.bit_width)

    # start the optimization main routine
    res = opt.opti_dnn(meta_data, hw_constraints)

    pprint.pprint(meta_data)

