#!/usr/bin/python2.7

# public library
import math
import numpy as np


# info for systolic array
A = 16.0      # systolic array dimension

# info for weights
K_w = 3.0       # kernel width
K_h = 3.0       # kernel height
K_d = 3.0       # kernel disparity
S = 1.0         # stride size

# input layer dimension
H = 512.0        # height of ofmap
W = 512.0        # width of ifmap
D = 128.0        # disparity dimension
Ci = 512.0      # channels for weights
Co = 512.0      # channels for ofmap

# memory bandwith number of bytes can be transferred.
B = 16.0/4

# on-chip buffer size
buffer_size = 1.0*1024.0*1024.0

# on-chip buffer partition
bufi_size = 0.3*1024.0*1024.0
bufo_size = 0.3*1024.0*1024.0
bufw_size = 0.4*1024.0*1024.0

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
# compute buffer utilization
def buffer_utilization(x):
    # buffer = ofmap + weights + ifmap
    return x[0]*x[1]*x[2]*x[3]+Ci*K_h*K_w*K_d*x[0]+Ci*(S*x[1]+2)*(S*x[2]+2)*(S*x[3]+2)


# set up hardware configuration
def setup_hardware3d(config):
    global A, B, buffer_size, bufi_size, bufo_size, bufw_size
    A = config[0]
    B = config[1]/4.0
    buffer_size = config[2]
    bufi_size = config[3]*buffer_size
    bufo_size = config[4]*buffer_size
    bufw_size = config[5]*buffer_size
    print("#CONFIG#",config)

def process_parameter(x, row_major, comp_bound):
    global res
    x = list(map(lambda i: math.floor(i), x))
    bound = "C"
    print(x)
    # make the tile size even for every batch
    c_0 = Co/math.ceil(Co/x[0])
    w_0 = W/math.ceil(W/x[1])
    h_0 = H/math.ceil(H/x[2])
    d_0 = D/math.ceil(D/x[3])
    # check the result
    print(c_0, w_0, h_0, d_0, Co/c_0, W/w_0, H/h_0, D/d_0)
    # compute the total number of elements needed to be updated 
    # if it is row-major.
    if row_major:
        # (ofmap + ifmap)*total_batch + (ofmap+weights)*Co/c_0
        total_transfer = (h_0*w_0*d_0*c_0+(S*h_0+2)*(S*w_0+2)*(S*d_0+2)*Ci)\
                            *H*W*D*Co/(h_0*w_0*d_0*c_0)\
                            +(h_0*w_0*d_0*c_0+K_h*K_w*K_d*Ci*c_0)*Co/c_0
    # compute the total number of elements needed to be updated 
    # if it is channel-major.
    else:
        # (ofmap + weights)*total_batch + (ofmap+ifmap)*(H*W)/(h_0*w_0)
        total_transfer = (h_0*w_0*d_0*c_0+K_h*K_w*K_d*Ci*c_0)*H*W*D*Co/(h_0*w_0*d_0*c_0)\
                        +(h_0*w_0*d_0*c_0+(S*h_0+2)*(S*w_0+2)*(S*d_0+2)*Ci)*H*W*D/(h_0*w_0*d_0)

    # compute the utilization of systolic array
    util_sys_arr = x[0]/(math.ceil(x[0]/A)*A) \
                        * x[1]*x[2]*x[3]/(math.ceil(x[1]*x[2]*x[3]/A)*A)

    # compute the utilization of systolic array
    util_buf = buffer_utilization([c_0, w_0, h_0, d_0])/buffer_size
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
    res.append([round(total_transfer, 0), round(total_cycle,0), util_sys_arr, util_buf, \
                [c_0, w_0, h_0, d_0], Co/c_0, W/w_0, H/h_0, D/d_0, bound])
    return

# the main optimization of compute-bound and row-major case;
def opti_buffer():
    # set the initial guess;
    x0 = [A, A]
    # first, let's find the number of kernel we can put into buffer.
    while (x0[0]+A)*K_h*K_w*K_d*Ci < bufw_size:
        x0[0] = x0[0]+A
    # set to be less than or equal to number of kernels
    x0[0] = min(x0[0], Co)

    # next, let's see how much ifmap can we fit into the buffer.
    while S*S*S*(x0[1]+A)*Ci < bufi_size and  x0[1] < W*H*D:
        x0[1] = x0[1]+A

    # no need to optimize the buffer for ofmap, because it is
    # bounded to ifmap.
    x = [x0[0], min(math.floor(math.sqrt(x0[1])), W), \
            min(math.floor(math.sqrt(x0[1])), H), 1]
    # set 
    x[-1] = min(math.floor(x0[1]/(x[1]*x[2])), D)

    process_parameter(x, False, False)
    process_parameter(x, False, True)
    process_parameter(x, True, False)
    process_parameter(x, True, True)

# optimize one layer
def optimize3d(layer_info):
    global W, H, D, Ci, Co, K_w, K_h, K_d, S
    del res[:]
    for item in layer_info[:6]:
        if item % 1 != 0:
            print("one input layer variable is not integer.")
            exit()
    # set up the new layer information
    (W, H, D, Ci, Co, K_w, K_h, K_d, S, _) = layer_info
    print("##[LAYER]##", W, H, D, Ci, Co, K_w, K_h, K_d)
    
    # both cases are possible;
    opti_buffer()

    if len(res) == 0:
        return None

    # choose the larger value as the bottleneck
    row_major_res = None
    if (res[0][1] < res[1][1]):
        row_major_res = res[1] 
    else: 
        row_major_res = res[0]

    # choose the larger value as the bottleneck
    channel_major_res = None
    if (res[2][1] < res[3][1]):
        channel_major_res = res[3] 
    else: 
        channel_major_res = res[2]

    # return the shortest value as the perferred compute ordering.
    ret = None
    if (row_major_res[1] < channel_major_res[1]):
        ret = row_major_res
    else:
        ret = channel_major_res

    return ret
