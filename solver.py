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
H = 128.0     # height
W = 256.0     # width
C = 512.0     # channel

# memory bandwith 
B = 2.0

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
#                 general process                       #
#########################################################

def process_parameter(x, row_major):

    res = [math.ceil(C/x[0]), C/math.ceil(C/x[0]), \
            math.ceil(W*H/x[1]), H*W/math.ceil(W*H/x[1])]

    print(math.ceil(C/x[0]), C/math.ceil(C/x[0]))
    print(math.ceil(W*H/x[1]), H*W/math.ceil(W*H/x[1]))

    x[0] = 16*math.floor(x[0]/16)
    x[1] = 16*math.floor(x[1]/16)

    print(math.ceil(C/x[0]), C/math.ceil(C/x[0]))
    print(math.ceil(W*H/x[1]), H*W/math.ceil(W*H/x[1]))

    if (row_major):
        total_transfer = (res[1]*res[3]+res[3]*C)*res[2]*res[0]\
                            +(res[1]*res[3]+K*K*C*res[3])*res[0]
    else:
        total_transfer = (res[1]*res[3]+K*K*C*res[1])*res[0]*res[2]\
                            +(res[1]*res[3]+res[3]*C)*res[2]

    print("total_transfer", total_transfer)
    return [res, total_transfer]

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

    process_parameter(solution.x, True)

def opti_comp_row_major():
    # set the initial guess;
    x0 = [A,A]
    # for row_major_constraint1
    con1 = {'type': 'ineq', 'fun': row_major_constraint}
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

    process_parameter(solution.x, True)

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

    process_parameter(solution.x, False)

def opti_comp_channel_major():
    # set the initial guess;
    x0 = [A,A]
    # for row_major_constraint1
    con1 = {'type': 'ineq', 'fun': channel_major_constraint}
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

    process_parameter(solution.x, False)


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

    # if it is possible to be memory-bound only;
    if (K*K*C)/(A*A) < B or B/((K*K*C)/(A*A) - B) > 1:
        opti_mem()  # only possible to be memory-bound;
    else:
        # both cases are possible;
        opti_mem()
        opti_comp()


if __name__== '__main__':
    optimizeLayer(H, W, C, C)


