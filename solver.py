#!/usr/bin/python2.7
# own module
# from memory_controller import *
# from systolic_array import *
# from onchip_buffer import *

# public library
import cv2 as cv
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

# info for systolic array
A = 16.0      # systolic array dimension

# info for weights
K = 3.0       # kernel size

# input layer dimension
H = 128.0      # height
W = 256.0     # width
C = 128.0     # channel

# memory bandwith 
B = 8.0

# buffer size
buffer_size = 2.0*1024.0*1024.0

# variables for optimization
# this two has been encodes as x[2]; = {c_0, h_0xw_0};
# c_0  # number of channels per batch;
# h_0xw_0 # size of tile per batch;

# calculate the latency for compute and memory;
# l_com = (K*K*c_0*h_0xw_0)/(R*R)
# # if row-major
# l_mem_r = B*(c_0*h_0xw_0 + C*h_0xw_0)
# # if channel-major
# l_mem_c = B*(c_0*h_0xw_0 + C*K*K*h_0xw_0)

#########################################################
#               general constraints                     #
#########################################################
# the low bound of buffer size
def buffer_constraint1(x):
    return x[0]*x[1]+C*K*K*x[0]+C*x[1]

# the upper bound of the buffer size
def buffer_constraint2(x):
    return buffer_size - (x[0]*x[1]+C*K*K*x[0]+C*x[1])

# make sure the process is always memory-bound;
# range : [0, +inf]
def mem_bound_constraint(x):
    return (B-K*K*C/(A*A))*x[0]+B*C

# make sure the process is always compute-bound;
# range : [0, +inf]
def comp_bound_constraint(x):
    return (K*K*C/(A*A)-B)*x[0]-B*C

#########################################################
#   row-major constraint solving obj and constraints    #
#########################################################

# the minimization objective of row-major memory-bound
def row_major_obj(x):
    # simplified from H*W*C*C/x[0] + K*K*C*C/x[1]
    return H*W/x[0] + K*K/x[1] 

# make sure the load for row-major is always less than 
# load for channel-major, range : [0, +inf]
def row_major_constraint(x):
    return K*K*x[0]-x[1];

def opti_mem_row_major():
    # set the initial guess;
    x0 = [A,A]
    # for row_major_constraint1
    con1 = {'type': 'ineq', 'fun': row_major_constraint}
    # for mem_bound_constraint
    con2 = {'type': 'ineq', 'fun': mem_bound_constraint}
    # for the buffer_constraint
    con3 = {'type': 'ineq', 'fun': buffer_constraint1}
    con4 = {'type': 'ineq', 'fun': buffer_constraint2}

    # summery all the bounds and constraints
    bnds = ((A, C), (A, H*W))
    cons= ([con1, con2, con3, con4])

    # call the external solver to solve the solution
    solution = minimize(row_major_obj,x0,method='SLSQP',\
                    bounds=bnds,constraints=cons)

    print("row major", solution.x, row_major_obj(solution.x))
    print(row_major_constraint(solution.x))
    print("buffer size", buffer_constraint1(solution.x))
    print(buffer_constraint2(solution.x))
    print(mem_bound_constraint(solution.x))

    

def opti_comp_row_major():
    # set the initial guess;
    x0 = [A,A]
    # for row_major_constraint1
    con1 = {'type': 'ineq', 'fun': row_major_constraint1}
    # for mem_bound_constraint
    con2 = {'type': 'ineq', 'fun': comp_bound_constraint}
    # for the buffer_constraint
    con3 = {'type': 'ineq', 'fun': buffer_constraint1}
    con4 = {'type': 'ineq', 'fun': buffer_constraint2}

    # summery all the bounds and constraints
    bnds = ((A, C), (A, H*W))
    cons= ([con1, con2, con3, con4])

    # call the external solver to solve the solution
    solution = minimize(row_major_obj,x0,method='SLSQP',\
                    bounds=bnds,constraints=cons)

    print("row major", solution.x, row_major_obj(solution.x))
    print(row_major_constraint(solution.x))
    print("buffer size", buffer_constraint1(solution.x))
    print(buffer_constraint2(solution.x))
    print(comp_bound_constraint(solution.x))

########################################################
# channel-major constraint solving obj and constraints #
########################################################

# the minimization objective of channel-major memory-bound
def channel_major_obj(x):
    # simplified from H*W*C/x[0] + K*K*C*C*W*H/x[1]
    return  1/x[0] + K*K*C/x[1]

# make sure the load for channel-major is always less than 
# load for row-major, range : [0, +inf]
def channel_major_constraint(x):
    return x[1]-K*K*x[0];

def opti_mem_channel_major():
    # set the initial guess;
    x0 = [A,A]
    # for row_major_constraint1
    con1 = {'type': 'ineq', 'fun': channel_major_constraint}
    # for mem_bound_constraint
    con2 = {'type': 'ineq', 'fun': mem_bound_constraint}
    # for the buffer_constraint
    con3 = {'type': 'ineq', 'fun': buffer_constraint1}
    con4 = {'type': 'ineq', 'fun': buffer_constraint2}

    # summery all the bounds and constraints
    bnds = ((A, C), (A, H*W))
    cons= ([con1, con2, con3, con4])

    # call the external solver to solve the solution
    solution = minimize(channel_major_obj,x0,method='SLSQP',\
                    bounds=bnds,constraints=cons)

    print("channel major",solution.x, channel_major_obj(solution.x))
    print(channel_major_constraint(solution.x))
    print("buffer size", buffer_constraint1(solution.x))
    print(buffer_constraint2(solution.x))
    print(mem_bound_constraint(solution.x))

def opti_comp_channel_major():
    # set the initial guess;
    x0 = [A,A]
    # for row_major_constraint1
    con1 = {'type': 'ineq', 'fun': channel_major_constraint1}
    # for mem_bound_constraint
    con2 = {'type': 'ineq', 'fun': comp_bound_constraint}
    # for the buffer_constraint
    con3 = {'type': 'ineq', 'fun': buffer_constraint1}
    con4 = {'type': 'ineq', 'fun': buffer_constraint2}

    # summery all the bounds and constraints
    bnds = ((A, C), (A, H*W))
    cons= ([con1, con2, con3, con4])

    # call the external solver to solve the solution
    solution = minimize(channel_major_obj,x0,method='SLSQP',\
                    bounds=bnds,constraints=cons)

    print("channel major",solution.x, channel_major_obj(solution.x))
    print(channel_major_constraint(solution.x))
    print("buffer size", buffer_constraint1(solution.x))
    print(buffer_constraint2(solution.x))
    print(comp_bound_constraint(solution.x))

def opti_mem():
    print("=================================")
    print("=======  Memory Bound  ==========")
    # optimization for row-major;
    opti_mem_row_major();
    # optimization for channel-major;
    opti_mem_channel_major();
    print("=================================\n")

def opti_comp():
    print("=================================")
    print("======  Compute Bound  ==========")
    # optimization for row-major;
    opti_comp_row_major();
    # optimization for channel-major;
    opti_comp_channel_major();
    print("=================================\n")


def optimizeLayer(height, width, channel, w_number):
    # info for systolic array
    A = 16.0      # systolic array dimension

    # info for weights
    K = 3.0       # kernel size

    # input layer dimension
    H = 128.0      # height
    W = 256.0     # width
    C = 128.0     # channel

    # memory bandwith 
    B = 8.0

    # buffer size
    buffer_size = 2.0*1024.0*1024.0

    # if it is possible to be memory-bound only;
    if (K*K*C)/(A*A) < B or B/((K*K*C)/(A*A) - B) > 1:
        opti_mem()  # only possible to be memory-bound;
    else:
        # both cases are possible;
        opti_mem()
        opti_comp()




    # plotMemoryTraffic(keys, w_batchs, total_traffics)
    # plotMemoryUsage(keys, total_mems)

   

# def plotMemoryUsage(keys, total_mems):
#     keys_log2 = np.log2(keys)


#     plt.rc('font', size=10)
#     ax1 = plt.figure(figsize=(6, 3)).add_subplot(111)
#     ax1.set_ylabel('Memory Size (in log2)', fontsize=12, fontweight='bold')
#     ax1.set_xscale('log', basex=2)
#     plt.setp(ax1.spines.values(), linewidth=2)

#     ax1.set_xlabel('Tile size (x*y)', fontsize=12, fontweight='bold')
#     # p1 = ax1.bar(keys_log2, total_mems, 0.4, align='center',color='#71985E',\
#     #     edgecolor=['k']*len(total_mems), linewidth=2, hatch="/");

#     p1 = ax1.plot(keys, total_mems, color='#71985E', linestyle='none',\
#             linewidth=2, markeredgecolor='k', marker='^', markersize=8);

#     plt.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
#     plt.subplots_adjust(left=0.10, bottom=0.20, right=0.95, top=0.9,
#                 wspace=0.2, hspace=0.2)

#     ax1.tick_params(axis="y",direction="in")
#     ax1.tick_params(axis="x",direction="in")
#     # ax1.set_ylim(0, 150)
#     plt.grid(color='grey', which='major', axis='y', linestyle='--') 
#     # plt.legend((p1[0], p2[0]), ('Batch', 'Traffic'), \
#     #         bbox_to_anchor=(0., 1.01, 1., .101), loc=3,
#     #         ncol=2, borderaxespad=0., frameon=False)
#     ax1.set_axisbelow(True)
    
#     plt.savefig("sched_mem.pdf");

# def plotMemoryTraffic(keys, w_batchs, total_traffics):
#     plt.rc('font', size=10)
#     ax1 = plt.figure(figsize=(6, 3)).add_subplot(111)
#     ax1.set_ylabel('Batch Size', fontsize=12, fontweight='bold')
#     ax1.set_xscale('log', basex=2)
#     plt.setp(ax1.spines.values(), linewidth=2)

#     ax2 = ax1.twinx()
#     ax2.set_ylabel('Memory Traffic in Log10', fontsize=12, fontweight='bold')
#     ax1.set_xlabel('Tile size (x*y)', fontsize=12, fontweight='bold')
#     # p1 = ax1.bar(keys, total_mem, 0.4, align='center',color='#71985E',\
#     #     edgecolor=['k']*len(ebs_axis_ls), linewidth=2, hatch="/");
    
#     p1 = ax1.plot(keys, w_batchs, color='#FFBF56', linestyle='--',\
#             linewidth=2, markeredgecolor='k', marker='^', markersize=8);
#     p2 = ax2.plot(keys, total_traffics, color='#8154D1', linestyle=':',\
#             linewidth=2, markeredgecolor='k', marker='o', markersize=8);

#     plt.subplots_adjust(left=0.1, bottom=0.20, right=0.9, top=0.9,
#                 wspace=0.2, hspace=0.2)

#     ax1.tick_params(axis="y",direction="in")
#     ax2.tick_params(axis="y",direction="in")
#     ax1.tick_params(axis="x",direction="in")
#     ax1.set_ylim(0, 150)
#     ax2.set_ylim(7.0, 10.0)
#     plt.grid(color='grey', which='major', axis='y', linestyle='--') 
#     plt.legend((p1[0], p2[0]), ('Batch', 'Traffic'), \
#             bbox_to_anchor=(0., 1.01, 1., .101), loc=3,
#             ncol=2, borderaxespad=0., frameon=False)
#     ax1.set_axisbelow(True)
    
#     plt.savefig("sched_traffic.pdf");
    
 


if __name__== '__main__':
    optimizeLayer(H, W, C, C)


