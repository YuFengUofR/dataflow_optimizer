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
K_d = 3.0       # kernel disparity
S = 1.0         # stride size

# input layer dimension
H = 128.0        # height of ofmap
W = 128.0        # width of ofmap
D = 128.0        # disparity ofmap 
Ci = 128.0      # channels for weights
Co = 128.0      # channels for ofmap

# memory bandwith number of bytes can be transferred.
B = 26.0/4

# on-chip buffer size
buffer_size = 1.0*1024.0*1024.0

# threshold for bounds
# if the constraint result is negative but within this threshold,
# it is still consider a valid result.
Threshold = 10.0

# array to store the result from the four different results
res = []

# variables for optimization
# this two has been encodes as x[3] = {c_0, h_0, w_0, d_0};
# c_0  # number of channels per batch;
# h_0, w_0, d_0 # the dimensions of tile per batch;
#

###############################################################
#                       general process                       #
###############################################################

def process_parameter(x, row_major, comp_bound):
    global res
    bound = "C"
    # make the tile size even for every batch
    c_0 = Co/math.ceil(Co/round(x[0]))
    w_0 = W/math.ceil(W/round(x[1]))
    h_0 = H/math.ceil(H/round(x[2]))
    d_0 = D/math.ceil(D/round(x[3]))
    # check the result
    print(c_0, w_0, h_0, d_0, Co/c_0, W/w_0, H/h_0, D/d_0)
    # compute the total number of elements needed to be updated 
    # if it is row-major.
    if row_major:
        # (ofmap + ifmap)*total_batch + (ofmap+weights)*Co/c_0
        total_transfer = (h_0*w_0*d_0*c_0+(S*h_0+2)*(S*w_0+2)*(S*d_0+2)*Ci) \
                            *(H*W*D*Co/(h_0*w_0*d_0*c_0)-Co/c_0)\
                            +(h_0*w_0*d_0*c_0+K_h*K_w*K_d*Ci*c_0)*Co/c_0
    # compute the total number of elements needed to be updated 
    # if it is channel-major.
    else:
        # (ofmap + weights)*total_batch + (ofmap+ifmap)*(H*W)/(h_0*w_0)
        total_transfer = (h_0*w_0*c_0+K_h*K_w*K_d*Ci*c_0) * \
                        (H*W*D*Co/(h_0*w_0*d_0*c_0)-H*W*D/(h_0*w_0*d_0)) \
                        +(h_0*w_0*d_0*c_0+(S*h_0+2)*(S*w_0+2)*(S*d_0+2)*Ci)*H*W*D/(h_0*w_0*d_0)

    # compute the utilization of systolic array
    util_sys_arr = x[0]/(math.ceil(round(x[0]/A, 1))*A) *\
                     x[1]*x[2]*x[3]/(math.ceil(round(x[1]*x[2]*x[3]/A, 1))*A)

    # compute the utilization of systolic array
    util_buf = buffer_constraint1([c_0, w_0, h_0, d_0])/buffer_size
    # calculate the amount of cycles of computing all elements.
    if comp_bound:
        bound = "C"
        total_cycle = (H*W*D*Co)*(Ci*K_h*K_w*K_d)/(A*A)/util_sys_arr 
    else:
        bound = "M"
        total_cycle = total_transfer/B

    # print(x[0],(math.ceil(x[0]/A)*A), x[1]*x[2], (math.ceil(x[1]*x[2]/A)*A))
    print("total_transfer", total_transfer, "total_cycle", total_cycle, \
        "systolic_array_utilization", util_sys_arr, "buffer_utilization", util_buf)
    res.append([int(total_transfer), int(total_cycle), util_sys_arr, \
                util_buf, x, Co/c_0, W/w_0, H/h_0, D/d_0, bound])
    return [int(total_transfer), int(total_cycle), util_sys_arr, \
                util_buf, x, Co/c_0, W/w_0, H/h_0, D/d_0, bound]

###############################################################
#                     general constraints                     #
###############################################################
# the low bound of buffer size;
# make sure the buffer utilization is always larger than 0
def buffer_constraint1(x):
    # buffer = ofmap + weights + ifmap
    return x[0]*x[1]*x[2]*x[3]+Ci*K_h*K_w*K_d*x[0]+Ci*(S*x[1]+2)*(S*x[2]+2)*(S*x[3]+2)

# the upper bound of the buffer size;
# make sure the buffer utilization is
# always smaller than buffer size;
def buffer_constraint2(x):
    return buffer_size - (x[0]*x[1]*x[2]*x[3] + \
            Ci*K_h*K_w*K_d*x[0] + Ci*(S*x[1]+2)*(S*x[2]+2)*(S*x[3]+2))


###############################################################
#       row-major constraint solving obj and constraints      #
###############################################################

# the minimization objective of row-major
# this objective is a simplified expression of 
# [h_0*w_0*d_0*c_0+(S*h_0+2)(S*w_0+2)(S*d_0+2)*Ci]*(H*W*D*Co)/(h_0*w_0*d_0*c_0)
# + [K^3*Ci+h_0*w_0*d_0*c_0]*Co/c_0
def row_major_mem_obj(x):
    return (x[0]*x[1]*x[2]*x[3] + (S*x[1]+2)*(S*x[2]+2)*(S*x[3]+2)*Ci) \
            *(H*W*D*Co/(x[0]*x[1]*x[2]*x[3])-Co/x[0]) \
            + (K_h*K_w*K_d*Ci)*Co/x[0] + x[1]*x[2]*x[3]*Co

def row_major_comp_obj(x):
    return H*W*D*Co/(x[1]*x[2]*x[3]*x[0])

# make sure the load for row-major is always less than 
# load for channel-major, range : [0, +inf]
def row_major_constraint(x):
    # simplified from K^3*Ci*c_0 > C*(S^3*h_0*w_0*d_0)
    return K_h*K_w*K_d*x[0] - (S*x[1]+2)*(S*x[2]+2)*(S*x[3]+2);

# make sure the process is always memory-bound;
# which is the latency for memory access is always 
# greater than lantecy of compute;
# (c_0*(h_0*w_0*d_0)+C*((S*h_0+2)*(S*w_0+2)*(S*d_0+2))/B >= (K^3*Ci/A^2)*c_0*w_0*d_0*h_0 
# range : [0, +inf]
def row_major_mem_bound_constraint(x):
    return (x[0]*x[1]*x[2]*x[3] + Ci*(S*x[1]+2)*(S*x[2]+2)*(S*x[3]+2))/B \
                - K_h*K_w*K_d*Ci/(A*A)*x[0]*x[1]*x[2]*x[3]

# the main optimization of memory-bound and row-major case; 
def opti_mem_row_major():
    # set the initial guess;
    x0 = [A, math.sqrt(A), math.sqrt(A), 1]
    # for row_major_constraint1
    con1 = {'type': 'ineq', 'fun': row_major_constraint}
    # for mem_bound_constraint
    con2 = {'type': 'ineq', 'fun': row_major_mem_bound_constraint}
    # for the buffer_constraint
    con3 = {'type': 'ineq', 'fun': buffer_constraint1}
    con4 = {'type': 'ineq', 'fun': buffer_constraint2}

    # summery all the bounds and constraints
    print((min(A, Co), Co), (min(math.floor(math.sqrt(A)), H), H), \
            (min(math.floor(math.sqrt(A)), W), W), (1, D))
    bnds = ((min(A, Co), Co), (min(math.floor(math.sqrt(A)), H), H), \
            (min(math.floor(math.sqrt(A)), W), W), (1, D))
    cons= ([con1, con2, con3, con4])

    # call the external solver to solve the solution
    solution = minimize(row_major_mem_obj, x0, method='SLSQP',\
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
        # passed = False
        print("buffer size", buffer_constraint1(solution.x), "is OVER limit!")
        # print("buffer constraint", buffer_constraint2(solution.x))
    if passed and row_major_mem_bound_constraint(solution.x) < -Threshold:
        passed = False
        print("row-major memory-bound", row_major_mem_bound_constraint(solution.x), \
            " no longer bounded!")
    
    if passed:
        print("Row-major memory-bound case PASSED!")
        return process_parameter(solution.x, True, False)
    else:
        return None

# make sure the process is always compute-bound;
# which is the latency for compute is always 
# greater than lantecy of memory access;
# (c_0*(h_0*w_0*d_0)+Ci*((S*h_0+2)*(S*w_0+2)*(S*d_0+2))/B <= (K^3*Ci/A^2)*c_0*w_0*h_0*d_0 
# range : [0, +inf]
def row_major_comp_bound_constraint(x):
    return K_h*K_w*K_d*Ci/(A*A)*x[0]*x[1]*x[2]*x[3] \
            - (x[0]*x[1]*x[2]*x[3] + Ci*(S*x[1]+2)*(S*x[2]+2)*(S*x[3]+2))/B

# the main optimization of compute-bound and row-major case;
def opti_comp_row_major():
    # set the initial guess;
    x0 = [min(A, Co), math.floor(math.sqrt(A)), math.floor(math.sqrt(A)), 1]
    # for row_major_constraint1
    con1 = {'type': 'ineq', 'fun': row_major_constraint}
    # for mem_bound_constraint
    con2 = {'type': 'ineq', 'fun': row_major_comp_bound_constraint}
    # for the buffer_constraint
    con3 = {'type': 'ineq', 'fun': buffer_constraint1}
    con4 = {'type': 'ineq', 'fun': buffer_constraint2}

    # summery all the bounds and constraints
    print((min(A, Co), Co), (min(math.floor(math.sqrt(A)), H), H), \
            (min(math.floor(math.sqrt(A)), W), W), (1, D))
    bnds = ((min(A, Co), Co), (min(math.floor(math.sqrt(A)), H), H), \
            (min(math.floor(math.sqrt(A)), W), W), (1, D))
    cons= ([con1, con2, con3, con4])

    # call the external solver to solve the solution
    solution = minimize(row_major_comp_obj, x0, method='SLSQP',\
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
        # passed = False
        print("buffer size", buffer_constraint1(solution.x), "is OVER limit!")
    if passed and row_major_comp_bound_constraint(solution.x) < -Threshold:
        passed = False
        print("Row-major compute-bound", row_major_comp_bound_constraint(solution.x), \
            " no longer bounded!")

    if passed:
        print("Row-major compute-bound case PASSED!")
        return process_parameter(solution.x, True, True)
    else:
        return None


###############################################################
#     channel-major constraint solving obj and constraints    #
###############################################################

# the minimization objective of channel-major
# this is the simplified expression of 
# (K^3*Ci*c_0+h_0*w_0*d_0*c_0)*(H*W*D*Co)/(h_0*w_0*d_0*c_0)
# + [(S*h_0+2)(S*w_0+2)(S*d_0+2)*Ci + h_0*w_0*d_0*c_0]*(H*W*D)/(h_0*w_0*d_0)
def channel_major_mem_obj(x):
    return  (K_h*K_w*K_d*Co)/(x[1]*x[2]*x[3]) + \
            (S*x[1]+2)*(S*x[2]+2)*(S*x[3]+2)/(x[1]*x[2]*x[3])

def channel_major_comp_obj(x):
    return H*W*D*Co/(x[1]*x[2]*x[0]*x[3])

# make sure the load for channel-major is always less than 
# load for row-major, range : [0, +inf]
def channel_major_constraint(x):
    # simplified from K^3*Ci*c_0 <= Ci*((S*h_0+2)*(S*w_0+2))
    return (S*x[1]+2)*(S*x[2]+2)*(S*x[3]+2) - K_h*K_w*K_d*x[0];

# make sure the process is always memory-bound;
# which is the latency for memory access is always 
# greater than lantecy of compute;
# c_0*(h_0*w_0+K^3*C)/B >= (K^3*C/A^2)*c_0*(h_0*w_0)
# range : [0, +inf]
def channel_major_mem_bound_constraint(x):
    return (x[1]*x[2]*x[3]+K_h*K_w*K_d*Ci)/B - K_h*K_w*K_d*Ci/(A*A)*x[1]*x[2]*x[3]

# the main optimization of memory-bound and channel-major case;
def opti_mem_channel_major():
    # set the initial guess;
    x0 = [A, math.sqrt(A), math.sqrt(A), 1]
    # for row_major_constraint1
    con1 = {'type': 'ineq', 'fun': channel_major_constraint}
    # for mem_bound_constraint
    con2 = {'type': 'ineq', 'fun': channel_major_mem_bound_constraint}
    # for the buffer_constraint
    con3 = {'type': 'ineq', 'fun': buffer_constraint1}
    con4 = {'type': 'ineq', 'fun': buffer_constraint2}

    # summery all the bounds and constraints
    print((min(A, Co), Co), (min(math.floor(math.sqrt(A)), H), H), \
            (min(math.floor(math.sqrt(A)), W), W), (1, D))
    bnds = ((min(A, Co), Co), (min(math.floor(math.sqrt(A)), H), H), \
            (min(math.floor(math.sqrt(A)), W), W), (1, D))
    cons= ([con1, con2, con3, con4])

    # call the external solver to solve the solution
    solution = minimize(channel_major_mem_obj, x0, method='SLSQP',\
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
        # passed = False
        print("buffer size", buffer_constraint1(solution.x), "is OVER limit!")
    if passed and channel_major_mem_bound_constraint(solution.x) < -Threshold:
        passed = False
        print("Channel-major memory-bound", channel_major_mem_bound_constraint(solution.x), \
            " no longer bounded!")

    if passed:
        print("Channel-major memory-bound case PASSED!")
        return process_parameter(solution.x, False, False)
    else:
        return None

# make sure the process is always memory-bound;
# which is the latency for memory access is always 
# greater than lantecy of compute;
# c_0*(h_0*w_0+K^3*C)/B >= (K^3*C/A^2)*c_0*(h_0*w_0*d_0) 
# range : [0, +inf]
def channel_major_comp_bound_constraint(x):
    return K_h*K_w*K_d*Co/(A*A)*x[1]*x[2]*x[3] - (x[1]*x[2]+K_h*K_w*K_d*Co)/B

# the main optimization of compute-bound and channel-major case;
def opti_comp_channel_major():
    # set the initial guess;
    x0 = [A, math.sqrt(A), math.sqrt(A), 1]
    # for row_major_constraint1
    con1 = {'type': 'ineq', 'fun': channel_major_constraint}
    # for mem_bound_constraint
    con2 = {'type': 'ineq', 'fun': channel_major_comp_bound_constraint}
    # for the buffer_constraint
    con3 = {'type': 'ineq', 'fun': buffer_constraint1}
    con4 = {'type': 'ineq', 'fun': buffer_constraint2}

    # summery all the bounds and constraints
    print((min(A, Co), Co), (min(math.floor(math.sqrt(A)), H), H), \
            (min(math.floor(math.sqrt(A)), W), W), (1, D))
    bnds = ((min(A, Co), Co), (min(math.floor(math.sqrt(A)), H), H), \
            (min(math.floor(math.sqrt(A)), W), W), (1, D))
    cons= ([con1, con2, con3, con4])

    # call the external solver to solve the solution
    solution = minimize(channel_major_comp_obj, x0, method='SLSQP',\
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
        # passed = False
        print("buffer size", buffer_constraint1(solution.x), "is OVER limit!")
    if passed and channel_major_comp_bound_constraint(solution.x) < -Threshold:
        passed = False
        print("Channel-major compute-bound", channel_major_comp_bound_constraint(solution.x), \
            " no longer bounded!")

    if passed:
        print("Channel-major compute-bound case PASSED!")
        return process_parameter(solution.x, False, True)
    else:
        return None


def opti_mem():
    print("=========================  Memory Bound  ==========================")
    # optimization for row-major;
    ret1 = opti_mem_row_major();
    # optimization for channel-major;
    ret2 = opti_mem_channel_major();
    print("\n")
    if ret1 == None and ret2 == None:
        return False
    else:
        return True

def opti_comp():
    print("=========================  Compute Bound  =========================")
    # optimization for row-major;
    ret1 = opti_comp_row_major();
    # optimization for channel-major;
    ret2 = opti_comp_channel_major();
    print("\n")
    if ret1 == None and ret2 == None:
        return False
    else:
        return True

def optimize3d(layer_info=None):
    global H, W, D, Ci, Co, K_w, K_h, K_d, S
    del res[:]
    for item in layer_info[:9]:
        if item % 1 != 0:
            print("one input layer variable is not integer.")
            exit()
    # set up the new layer information
    (W, H, D, Ci, Co, K_w, K_h, K_d, S, _) = layer_info
    print("##[LAYER]##", W, H, D, Ci, Co, K_w, K_h, K_d)
    
    # both cases are possible;
    # opti_mem()
    ret = opti_comp()

    if ret is False:
        opti_mem()

    if len(res) == 0:
        return None

    ret  = list(res[0])

    for item in res:
        if ret[1] > item[1]:
            ret = list(item)
        if ret[1] == item[1] and ret[0] > item[0]:
            ret = list(item)

    return ret

def setup_hardware3d(config):
    global A, B, buffer_size
    A = config[0]
    B = config[1]/4.0
    buffer_size = config[2]


# optimize()
