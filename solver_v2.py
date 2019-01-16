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
H = 512.0        # height of ofmap
W = 512.0        # width of ifmap
Ci = 512.0      # channels for weights
Co = 512.0      # channels for ofmap

# memory bandwith number of bytes can be trasferred.
B = 2.0/4

# on-chip buffer size
buffer_size = 1.0*1024.0*1024.0

# variables for optimization
# this two has been encodes as x[3] = {c_0, h_0, w_0};
# c_0  # number of channels per batch;
# h_0xw_0 # size of tile per batch;

# calculate the latency for compute and memory;
# l_com = (K*K*c_0*h_0*w_0)/(R*R)
# # if row-major
# l_mem_r = (c_0*h_0*w_0 + C*h_0*w_0)/B
# # if channel-major
# l_mem_c = (c_0*h_0*w_0 + C*K*K*h_0*w_0)/B

###############################################################
#                       general process                       #
###############################################################

def process_parameter(x, row_major, comp_bound):

    c_0 = Co/math.ceil(Co/x[0])
    w_0 = W/math.ceil(W/x[1])
    h_0 = H/math.ceil(H/x[2])

    print(c_0, w_0, h_0, Co/c_0, W/w_0, H/h_0)

    if (row_major):
        total_transfer = (h_0*w_0*c_0+(h_0+2)*(w_0+2)*Ci)*H*W*Co/(h_0*w_0*c_0)\
                            +(h_0*w_0*c_0+K*K*Ci*c_0)*Co/c_0
    else:
        total_transfer = (h_0*w_0*c_0+K*K*Ci*c_0)*H*W*Co/(h_0*w_0*c_0)\
                            +(h_0*w_0*c_0+(h_0+2)*(w_0+2)*Ci)*H*W/(h_0*w_0)

    if comp_bound:
        total_cycle = (H*W*Co)*(Ci*K*K)/(A*A)
    else:
        total_cycle = total_transfer/B

    print("total_transfer", total_transfer)
    print("total_cycle", total_cycle)
    return


# this function is to verifer if a given hardware
# configuration is able to realize in given hardware 
# constraints. 
# return the result and total 

# def verifier(x, row_major):

###############################################################
#                     general constraints                     #
###############################################################
# the low bound of buffer size;
# make sure the buffer utilization is always larger than 0
def buffer_constraint1(x):
    # buffer = ofmap + weights + ifmap
    return x[0]*x[1]*x[2]+Ci*K*K*x[0]+Ci*(x[1]+2)*(x[2]+2)

# the upper bound of the buffer size;
# make sure the buffer utilization is
# always smaller than buffer size;
def buffer_constraint2(x):
    return buffer_size - (x[0]*x[1]*x[2]+Ci*K*K*x[0]+Ci*(x[1]+2)*(x[2]+2))


###############################################################
#       row-major constraint solving obj and constraints      #
###############################################################

# the minimization objective of row-major
# this objective is a simplified expression of 
# [h_0*w_0*c_0+(h_0+2)(w_0+2)*Ci]*(H*W*Co)/(h_0*w_0*c_0)
# + [K^2*Ci+h_0*w_0*c_0]*C/c_0
def row_major_obj(x):
    return H*W*Ci/x[0]*(1+2*(x[1]+x[2])/(x[1]*x[2]))+x[1]*x[2]/x[0]

# make sure the load for row-major is always less than 
# load for channel-major, range : [0, +inf]
def row_major_constraint(x):
    # simplified from K^2*C*c_0 > C*(h_0*w_0)
    return K*K*x[0] - x[1]*x[2];

# make sure the process is always memory-bound;
# which is the latency for memory access is always 
# greater than lantecy of compute;
# (c_0+C)*(h_0*w_0)/B >= (K^2*C/A^2)*c_0*w_0*h_0 
# range : [0, +inf]
def row_major_mem_bound_constraint(x):
    return (x[0]+Ci)/B - K*K*Ci/(A*A)*x[0]

# the main optimization of memory-bound and row-major case; 
def opti_mem_row_major():
    # set the initial guess;
    x0 = [A, math.sqrt(A), math.sqrt(A)]
    # for row_major_constraint1
    con1 = {'type': 'ineq', 'fun': row_major_constraint}
    # for mem_bound_constraint
    con2 = {'type': 'ineq', 'fun': row_major_mem_bound_constraint}
    # for the buffer_constraint
    con3 = {'type': 'ineq', 'fun': buffer_constraint1}
    con4 = {'type': 'ineq', 'fun': buffer_constraint2}

    # summery all the bounds and constraints
    bnds = ((A, Co), (math.sqrt(A), H), (math.sqrt(A), W))
    cons= ([con1, con2, con3, con4])

    # call the external solver to solve the solution
    solution = minimize(row_major_obj, x0, method='SLSQP',\
                    bounds=bnds,constraints=cons)

    print("row major", solution.x, row_major_obj(solution.x))
    print("row major constraint", row_major_constraint(solution.x))
    print("buffer size", buffer_constraint1(solution.x))
    print("buffer constraint", buffer_constraint2(solution.x))
    print(row_major_mem_bound_constraint(solution.x))

    process_parameter(solution.x, True, False)

# make sure the process is always compute-bound;
# which is the latency for compute is always 
# greater than lantecy of memory access;
# (c_0+C)*(h_0*w_0)/B <= (K^2*C/A^2)*c_0*h_0*w_0 
# range : [0, +inf]
def row_major_comp_bound_constraint(x):
    return K*K*Ci/(A*A)*x[0]-(Ci+x[0])/B

# the main optimization of compute-bound and row-major case;
def opti_comp_row_major():
    # set the initial guess;
    x0 = [A, math.sqrt(A), math.sqrt(A)]
    # for row_major_constraint1
    con1 = {'type': 'ineq', 'fun': row_major_constraint}
    # for mem_bound_constraint
    con2 = {'type': 'ineq', 'fun': row_major_comp_bound_constraint}
    # for the buffer_constraint
    con3 = {'type': 'ineq', 'fun': buffer_constraint1}
    con4 = {'type': 'ineq', 'fun': buffer_constraint2}

    # summery all the bounds and constraints
    bnds = ((A, Co), (math.sqrt(A), H), (math.sqrt(A), W))
    cons= ([con1, con2, con3, con4])

    # call the external solver to solve the solution
    solution = minimize(row_major_obj,x0,method='SLSQP',\
                    bounds=bnds,constraints=cons)

    print("row major", solution.x, row_major_obj(solution.x))
    print("row major constraint", row_major_constraint(solution.x))
    print("buffer size", buffer_constraint1(solution.x))
    print("buffer constraint", buffer_constraint2(solution.x))
    print(row_major_comp_bound_constraint(solution.x))

    process_parameter(solution.x, True, True)


###############################################################
#     channel-major constraint solving obj and constraints    #
###############################################################

# the minimization objective of channel-major
# this is the simplified expression of 
# (K^2*Ci*c_0+h_0*w_0*c_0)*(H*W*Co)/(h_0*w_0*c_0)
# + [(h_0+2)(w_0+2)*Ci + h_0*w_0*c_0]*(H*W)/(h_0*w_0)
def channel_major_obj(x):
    return  (K*K*Ci*Co)/(x[1]*x[2])+2*(x[1]+x[2])*Co/(x[1]*x[2])+1/x[0]

# make sure the load for channel-major is always less than 
# load for row-major, range : [0, +inf]
def channel_major_constraint(x):
    # simplified from K^2*C*c_0 <= C*(h_0*w_0)
    return x[1]*x[2] - K*K*x[0];

# make sure the process is always memory-bound;
# which is the latency for memory access is always 
# greater than lantecy of compute;
# c_0*(h_0*w_0+K^2*C)/B >= (K^2*C/A^2)*c_0*(h_0*w_0)
# range : [0, +inf]
def channel_major_mem_bound_constraint(x):
    return (x[1]*x[2]+K*K*Ci)/B - K*K*Ci/(A*A)*x[1]*x[2]

# the main optimization of memory-bound and channel-major case;
def opti_mem_channel_major():
    # set the initial guess;
    x0 = [A, math.sqrt(A), math.sqrt(A)]
    # for row_major_constraint1
    con1 = {'type': 'ineq', 'fun': channel_major_constraint}
    # for mem_bound_constraint
    con2 = {'type': 'ineq', 'fun': channel_major_mem_bound_constraint}
    # for the buffer_constraint
    con3 = {'type': 'ineq', 'fun': buffer_constraint1}
    con4 = {'type': 'ineq', 'fun': buffer_constraint2}

    # summery all the bounds and constraints
    bnds = ((A, Co), (math.sqrt(A), H), (math.sqrt(A), W))
    cons= ([con1, con2, con3, con4])

    # call the external solver to solve the solution
    solution = minimize(channel_major_obj,x0,method='SLSQP',\
                    bounds=bnds,constraints=cons)

    print("channel major",solution.x, channel_major_obj(solution.x))
    print("channel major constraint", channel_major_constraint(solution.x))
    print("buffer size", buffer_constraint1(solution.x))
    print("buffer constraint", buffer_constraint2(solution.x))
    print(channel_major_mem_bound_constraint(solution.x))

    process_parameter(solution.x, False, False)

# make sure the process is always memory-bound;
# which is the latency for memory access is always 
# greater than lantecy of compute;
# c_0*(h_0*w_0+K^2*C)/B >= (K^2*C/A^2)*c_0*(h_0*w_0) 
# range : [0, +inf]
def channel_major_comp_bound_constraint(x):
    return K*K*Co/(A*A)*x[1]*x[2] - (x[1]*x[2]+K*K*Co)/B

# the main optimization of compute-bound and channel-major case;
def opti_comp_channel_major():
    # set the initial guess;
    x0 = [A, math.sqrt(A), math.sqrt(A)]
    # for row_major_constraint1
    con1 = {'type': 'ineq', 'fun': channel_major_constraint}
    # for mem_bound_constraint
    con2 = {'type': 'ineq', 'fun': channel_major_comp_bound_constraint}
    # for the buffer_constraint
    con3 = {'type': 'ineq', 'fun': buffer_constraint1}
    con4 = {'type': 'ineq', 'fun': buffer_constraint2}

    # summery all the bounds and constraints
    bnds = ((A, Co), (math.sqrt(A), H), (math.sqrt(A), W))
    cons= ([con1, con2, con3, con4])

    # call the external solver to solve the solution
    solution = minimize(channel_major_obj,x0,method='SLSQP',\
                    bounds=bnds,constraints=cons)

    print("channel major",solution.x, channel_major_obj(solution.x))
    print("channel major constraint", channel_major_constraint(solution.x))
    print("buffer size", buffer_constraint1(solution.x))
    print("buffer constraint", buffer_constraint2(solution.x))
    print(channel_major_comp_bound_constraint(solution.x))

    process_parameter(solution.x, False, True)


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
    if (K*K*Ci)/(A*A) < B or B/((K*K*Ci)/(A*A) - B) > 1:
        opti_mem()  # only possible to be memory-bound;
    else:
        # both cases are possible;
        opti_mem()
        opti_comp()


if __name__== '__main__':
    optimizeLayer(H, W, Ci, Co)


