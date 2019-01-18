#!/usr/bin/python2.7

# public library
import cv2 as cv
import math
import numpy as np
from scipy.optimize import minimize


# info for systolic array
A = 16.0      # systolic array dimension

# info for weights
K_w = 3.0       # kernel width
K_h = 3.0       # kernel height
S = 1.0         # stride size

# input layer dimension
H = 512.0        # height of ofmap
W = 512.0        # width of ifmap
Ci = 512.0      # channels for weights
Co = 512.0      # channels for ofmap

# memory bandwith number of bytes can be trasferred.
B = 4.0/4

# on-chip buffer size
buffer_size = 1.0*1024.0*1024.0

# threshold for bounds
# if the constraint result is negative but within this threshold,
# it is still consider a valid result.
Threshold = 10.0

# array to store the result from the four different results
res = []

# variables for optimization
# this two has been encodes as x[3] = {c_0, h_0, w_0};
# c_0  # number of channels per batch;
# h_0xw_0 # size of tile per batch;

# calculate the latency for compute and memory;
# l_com = (K_h*K_w*c_0*h_0*w_0)/(R*R)
# # if row-major
# l_mem_r = (c_0*h_0*w_0 + C*(h_0+2)*(w_0+2))/B
# # if channel-major
# l_mem_c = (c_0*h_0*w_0 + C*K_h*K_w*c_0)/B

###############################################################
#                       general process                       #
###############################################################

def process_parameter(x, row_major, comp_bound):
    x = list(map(lambda i: math.floor(i), x))
    # make the tile size even for every batch
    c_0 = Co/math.ceil(Co/x[0])
    w_0 = W/math.ceil(W/x[1])
    h_0 = H/math.ceil(H/x[2])
    # check the result
    print(c_0, w_0, h_0, Co/c_0, W/w_0, H/h_0)
    # compute the total number of elements needed to be updated 
    # if it is row-major.
    if row_major:
        # (ofmap + ifmap)*total_batch + (ofmap+weights)*Co/c_0
        total_transfer = (h_0*w_0*c_0+(S*h_0+2)*(S*w_0+2)*Ci)*H*W*Co/(h_0*w_0*c_0)\
                            +(h_0*w_0*c_0+K_h*K_w*Ci*c_0)*Co/c_0
    # compute the total number of elements needed to be updated 
    # if it is channel-major.
    else:
        # (ofmap + weights)*total_batch + (ofmap+ifmap)*(H*W)/(h_0*w_0)
        total_transfer = (h_0*w_0*c_0+K_h*K_w*Ci*c_0)*H*W*Co/(h_0*w_0*c_0)\
                            +(h_0*w_0*c_0+(S*h_0+2)*(S*w_0+2)*Ci)*H*W/(h_0*w_0)

    # calculate the amount of cycles of computing all elements.
    if comp_bound:
        total_cycle = (H*W*Co)*(Ci*K_h*K_w)/(A*A)
    else:
        total_cycle = total_transfer/B

    # compute the utilization of systolic array
    util_sys_arr = x[0]/(math.ceil(x[0]/A)*A) \
                        * x[1]*x[2]/(math.ceil(x[1]*x[2]/A)*A)
    print(x[0],(math.ceil(x[0]/A)*A), x[1]*x[2], (math.ceil(x[1]*x[2]/A)*A))

    print("total_transfer", total_transfer, "total_cycle", total_cycle, "systolic_array_utilization", util_sys_arr)
    res.append((total_transfer, total_cycle, util_sys_arr, Co/c_0, W/w_0, H/h_0))
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
    return x[0]*x[1]*x[2]+Ci*K_h*K_w*x[0]+Ci*(x[1]+2)*(x[2]+2)

# the upper bound of the buffer size;
# make sure the buffer utilization is
# always smaller than buffer size;
def buffer_constraint2(x):
    return buffer_size - (x[0]*x[1]*x[2]+Ci*K_h*K_w*x[0]+Ci*(x[1]+2)*(x[2]+2))


###############################################################
#       row-major constraint solving obj and constraints      #
###############################################################

# the minimization objective of row-major
# this objective is a simplified expression of 
# [h_0*w_0*c_0+(h_0+2)(w_0+2)*Ci]*(H*W*Co)/(h_0*w_0*c_0)
# + [K^2*Ci+h_0*w_0*c_0]*C/c_0
# this expression can be finally reduce to:
#   (H*W*Co/c_0 + 2(h_0+w_0)Ci*H*W*Co/(h_0*w_0*c_0)+h_0*w_0*Co/c_0
def row_major_obj(x):
    return H*W*Co/x[0]*(1+2*(x[1]+x[2])*Ci/(x[1]*x[2]))+x[1]*x[2]/x[0]

# make sure the load for row-major is always less than 
# load for channel-major, range : [0, +inf]
def row_major_constraint(x):
    # simplified from K^2*C*c_0 > C*(S^2*h_0*w_0)
    return K_h*K_w*x[0] - (S*x[1]+2)*(S*x[2]+2);

# make sure the process is always memory-bound;
# which is the latency for memory access is always 
# greater than lantecy of compute;
# (c_0*(h_0*w_0)+C*((S*h_0+2)*(S*w_0+2))/B >= (K^2*C/A^2)*c_0*w_0*h_0 
# range : [0, +inf]
def row_major_mem_bound_constraint(x):
    return (x[0]*x[1]*x[2] + Ci*(S*x[1]+2)*(S*x[2]+2))/B \
                - K_h*K_w*Ci/(A*A)*x[0]

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

    passed = True
    if np.any(np.isnan(solution.x)):
        passed = False
        print("Solution with NaN, abort!")
    # check the validation
    if passed and row_major_constraint(solution.x) < -Threshold:
        passed = False
        print("row major constraint", row_major_constraint(solution.x), "NOT PASSED.")
    if passed and buffer_constraint2(solution.x) < -Threshold:
        passed = False
        print("buffer size", buffer_constraint1(solution.x), "is OVER limit!")
        # print("buffer constraint", buffer_constraint2(solution.x))
    if passed and row_major_mem_bound_constraint(solution.x) < -Threshold:
        passed = False
        print("row-major memory-bound", row_major_mem_bound_constraint(solution.x), \
            " no longer bounded!")
    
    if passed:
        print("Row-major memory-bound case PASSED!")
        process_parameter(solution.x, True, False)
    else:
        return None

# make sure the process is always compute-bound;
# which is the latency for compute is always 
# greater than lantecy of memory access;
# (c_0*(h_0*w_0)+C*((S*h_0+2)*(S*w_0+2))/B <= (K^2*C/A^2)*c_0*w_0*h_0 
# range : [0, +inf]
def row_major_comp_bound_constraint(x):
    return K_h*K_w*Ci/(A*A)*x[0] \
            - (x[0]*x[1]*x[2] + Ci*(S*x[1]+2)*(S*x[2]+2))/B
                

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
    solution = minimize(row_major_obj, x0, method='SLSQP',\
                    bounds=bnds, constraints=cons)

    passed = True
    if np.any(np.isnan(solution.x)):
        passed = False
        print("Solution with NaN, abort!")
    # check the validation
    if passed and row_major_constraint(solution.x) < -Threshold:
        passed = False
        print("row major constraint", row_major_constraint(solution.x), "NOT PASSED.")
    if passed and buffer_constraint2(solution.x) < -Threshold:
        passed = False
        print("buffer size", buffer_constraint1(solution.x), "is OVER limit!")
    if passed and row_major_comp_bound_constraint(solution.x) < -Threshold:
        passed = False
        print("Row-major compute-bound", row_major_comp_bound_constraint(solution.x), \
            " no longer bounded!")

    if passed:
        print("Row-major compute-bound case PASSED!")
        process_parameter(solution.x, True, True)
    else:
        return None


###############################################################
#     channel-major constraint solving obj and constraints    #
###############################################################

# the minimization objective of channel-major
# this is the simplified expression of 
# (K^2*Ci*c_0+h_0*w_0*c_0)*(H*W*Co)/(h_0*w_0*c_0)
# + [(h_0+2)(w_0+2)*Ci + h_0*w_0*c_0]*(H*W)/(h_0*w_0)
def channel_major_obj(x):
    return  (K_h*K_w*Ci*Co)/(x[1]*x[2])+2*(x[1]+x[2])*Co/(x[1]*x[2])+1/x[0]

# make sure the load for channel-major is always less than 
# load for row-major, range : [0, +inf]
def channel_major_constraint(x):
    # simplified from K^2*C*c_0 <= C*((S*h_0+2)*(S*w_0+2))
    return (S*x[1]+2)*(S*x[2]+2) - K_h*K_w*x[0];

# make sure the process is always memory-bound;
# which is the latency for memory access is always 
# greater than lantecy of compute;
# c_0*(h_0*w_0+K^2*C)/B >= (K^2*C/A^2)*c_0*(h_0*w_0)
# range : [0, +inf]
def channel_major_mem_bound_constraint(x):
    return (x[1]*x[2]+K_h*K_w*Ci)/B - K_h*K_w*Ci/(A*A)*x[1]*x[2]

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
    solution = minimize(channel_major_obj, x0, method='SLSQP',\
                    bounds=bnds, constraints=cons)

    passed = True
    if np.any(np.isnan(solution.x)):
        passed = False
        print("Solution with NaN, abort!")
    # check the validation
    if passed and channel_major_constraint(solution.x) < -Threshold:
        passed = False
        print("channel major constraint", channel_major_constraint(solution.x), "NOT PASSED.")
    if passed and buffer_constraint2(solution.x) < -Threshold:
        passed = False
        print("buffer size", buffer_constraint1(solution.x), "is OVER limit!")
    if passed and channel_major_mem_bound_constraint(solution.x) < -Threshold:
        passed = False
        print("Channel-major memory-bound", channel_major_mem_bound_constraint(solution.x), \
            " no longer bounded!")

    if passed:
        print("Channel-major memory-bound case PASSED!")
        process_parameter(solution.x, False, False)
    else:
        return None

# make sure the process is always memory-bound;
# which is the latency for memory access is always 
# greater than lantecy of compute;
# c_0*(h_0*w_0+K^2*C)/B >= (K^2*C/A^2)*c_0*(h_0*w_0) 
# range : [0, +inf]
def channel_major_comp_bound_constraint(x):
    return K_h*K_w*Co/(A*A)*x[1]*x[2] - (x[1]*x[2]+K_h*K_w*Co)/B

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
    solution = minimize(channel_major_obj, x0, method='SLSQP',\
                    bounds=bnds, constraints=cons)

    passed = True
    if np.any(np.isnan(solution.x)):
        passed = False
        print("Solution with NaN, abort!")
    # check the validation
    if passed and channel_major_constraint(solution.x) < -Threshold:
        passed = False
        print("channel major constraint", channel_major_constraint(solution.x), "NOT PASSED.")
    if passed and buffer_constraint2(solution.x) < -Threshold:
        passed = False
        print("buffer size", buffer_constraint1(solution.x), "is OVER limit!")
    if passed and channel_major_comp_bound_constraint(solution.x) < -Threshold:
        passed = False
        print("Channel-major compute-bound", channel_major_comp_bound_constraint(solution.x), \
            " no longer bounded!")

    if passed:
        print("Channel-major compute-bound case PASSED!")
        process_parameter(solution.x, False, True)
    else:
        return None


def opti_mem():
    print("=========================  Memory Bound  ==========================")
    # optimization for row-major;
    opti_mem_row_major();
    # optimization for channel-major;
    opti_mem_channel_major();
    print("\n")

def opti_comp():
    print("=========================  Compute Bound  =========================")
    # optimization for row-major;
    opti_comp_row_major();
    # optimization for channel-major;
    opti_comp_channel_major();
    print("\n")


def optimize(layer_info):
    global H, W, Ci, Co, K_w, K_h, S
    del res[:]
    for item in layer_info[:6]:
        if item % 1 != 0:
            print("one input layer variable is not integer.")
            exit()
    # set up the new layer information
    (W, H, Ci, Co, K_w, K_h, S, _) = layer_info
    print("##[LAYER]##", W, H, Ci, Co, K_w, K_h)
    
    # both cases are possible;
    opti_mem()
    opti_comp()

    if len(res) == 0:
        return None

    ret  = res[0]

    for item in res:
        if ret[1] > item[1]:
            ret = item

    return item

def setup_hardware(config):
    global A, B, buffer_size
    A = config[0]
    B = config[1]/4
    buffer_size = config[2]



