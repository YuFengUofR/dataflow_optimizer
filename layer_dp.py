#!/usr/bin/python2.7

# public library
import math
import numpy as np


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

# memory bandwith number of bytes can be transferred.
B = 16.0/4

# on-chip buffer size
bufi_size = 0.3*1024.0*1024.0*2/4
bufo_size = 0.3*1024.0*1024.0*2/4
bufw_size = 0.4*1024.0*1024.0*2/4

buffer_size = 2.0*1024.0*1024.0/4.0

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
    return x[0]*x[1]*x[2]+Ci*K_h*K_w*x[0]+Ci*(S*x[1]+2)*(S*x[2]+2)


# set up hardware configuration
def setup_hardware(config):
    global A, B, buffer_size, bufi_size, bufo_size, bufw_size

    A = config[0]
    B = config[1]/4.0
    buffer_size = config[2]
    # on-chip buffer partition
    bufi_size = 0.3*buffer_size
    bufo_size = 0.3*buffer_size
    bufw_size = 0.4*buffer_size

def process_parameter(x, row_major, comp_bound):
    global res
    x = list(map(lambda i: math.floor(i), x))
    bound = "C"
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

    # compute the utilization of systolic array
    util_sys_arr = x[0]/(math.ceil(x[0]/A)*A) \
                        * x[1]*x[2]/(math.ceil(x[1]*x[2]/A)*A)

    # compute the utilization of systolic array
    util_buf = buffer_utilization([c_0, w_0, h_0])/buffer_size
    # calculate the amount of cycles of computing all elements.
    if comp_bound:
        bound = "C"
        total_cycle = (H*W*Co)*(Ci*K_h*K_w)/(A*A)/util_sys_arr 
    else:
        bound = "M"
        total_cycle = total_transfer/B

    # print(x[0],(math.ceil(x[0]/A)*A), x[1]*x[2], (math.ceil(x[1]*x[2]/A)*A))
    print("total_transfer", total_transfer, "total_cycle", total_cycle, \
        "systolic_array_utilization", util_sys_arr, "buffer_utilization", util_buf)
    res.append([int(total_transfer), int(total_cycle), util_sys_arr, util_buf,\
         x, Co/c_0, W/w_0, H/h_0, bound])

    return

# the main optimization of compute-bound and row-major case;
def opti_buffer():
    global A, B, buffer_size, bufi_size, bufo_size, bufw_size
    # set the initial guess;
    x0 = [A, A]
    # first, let's find the number of kernel we can put into buffer.
    while (x0[0]+A)*K_h*K_w*Ci < bufw_size:
        x0[0] = x0[0]+A

    # next let's see how much ifmap can we fit into the buffer.
    while S*S*(x0[1]+A)*Ci < bufi_size:
        x0[1] = x0[1]+A

    # no need to optimize the buffer for ofmap, because it is
    # bounded ifmap.

    x = [x0[0], math.sqrt(x0[1]), math.sqrt(x0[1])]
    process_parameter(x, False, False)
    process_parameter(x, False, True)
    process_parameter(x, True, False)
    process_parameter(x, True, True)


# the main optimization of compute-bound and row-major case;
# subs <- a list of (width(0), height(1), in_channel(2), out_channel(3),
#                   kenrel_width(4), kernel_height(5), stride(6), Deconv?(7))
def opti_deconv_buffer(subs):
    global A, B, buffer_size, bufi_size, bufo_size, bufw_size
    # record the remaining number of out_channels
    num_subs = list([sub[3] for sub in subs])

    # set the initial guess;
    Area = A
    # next let's see how much ifmap can we fit into the buffer.
    while S*S*(Area+A)*Ci < bufi_size and Area < W*H:
        Area += A

    w_0 = W/math.ceil(W/min(math.sqrt(Area), W))
    h_0 = H/math.ceil(H/min(math.sqrt(Area), H))
    # print("[AERA]", Area, w_0, h_0)

    curr_bufw = 1.0

    # the data needed to load for ifmap, consider stride 1
    total_transfer = 0 # (h_0+2)*(w_0+2)*Ci
    total_cycle = 0.0

    cnt_round = 0
    buf_util = []
    while curr_bufw > 0.0:
        # print("[round]", cnt_round)
        cnt_round += 1
        inx = 0
        curr_bufw = 0.0
        curr_bufo = 0.0
        cnt_arr = []
        for sub in subs:
            cnt = 0
            # first, let's find the number of kernel we can put into buffer.
            while (curr_bufw + A*sub[4]*sub[5]*Ci) < bufw_size \
                                                and cnt < num_subs[inx]:
                # add additional weight into current weight buffer
                curr_bufw += A*sub[4]*sub[5]*Ci
                # add additional output into current output buffer
                curr_bufo += w_0*h_0*A
                # add additional computation
                total_cycle += sub[4]*sub[5]*Ci*math.ceil(w_0*h_0/A)
                # 
                cnt += A

            # update the index and cnt_arr
            inx += 1
            cnt_arr.append(cnt)

        # subtract the value out of num_subs
        num_subs = np.subtract(num_subs, cnt_arr)
        # end of the loop, check cnt_arr value
        # print("sub_arr", cnt_arr, "num_subs", num_subs)
        # add additional data transfer
        if curr_bufo == 0:
            break
        total_transfer += (curr_bufw + curr_bufo)
        # print("bufo util", curr_bufo/bufo_size, "bufw util", curr_bufw/bufw_size)
        buf_util.append((curr_bufw + curr_bufo + (h_0+2)*(w_0+2)*Ci)/buffer_size)

    if total_transfer/B > total_cycle:
        total_cycle = total_transfer/B
    

    batch = math.ceil(W/w_0*H/h_0)

    for val in num_subs:
        if val > 0:
            return [total_transfer*batch, total_cycle*batch, 0, 0, False]

    # utilizaition of systolic array
    total_comp = 0
    for sub in subs:
        total_comp += sub[4]*sub[5]*Ci*W*H*Co

    util_sys_arr = total_comp/(total_cycle*A*A*batch)

    util_buf = 0
    if len(buf_util) != 0:
        util_buf = np.mean(buf_util)

    # check whether row-major is benefitial or channel-major
    if total_transfer < H*W*Ci:
        return [round(total_transfer*batch+H*W*Ci,1), round(total_cycle*batch,1), \
            util_sys_arr, util_buf, True]
    else:
        return [round(total_transfer+H*W*Ci*batch,1), round(total_cycle*batch,1), \
            util_sys_arr, util_buf, True]


# optimize deconv layer
def optimize_deconv(subs):
    global H, W, Ci, Co, K_w, K_h, S, buffer_size, \
            bufi_size, bufo_size, bufw_size

    # set up the new layer information

    for i in range(4):
        (W, H, Ci, Co, K_w, K_h, S, _) = subs[i]
        # print("##[LAYER%d]##" % (i), W, H, Ci, Co, K_w, K_h)

    best_res = None
    for i in range(1, 50):
        # set up the configuration
        bufi_size = buffer_size*(1.0*i/100.0)
        bufo_size = buffer_size*(1.0*i/100.0)
        bufw_size = buffer_size*((100.0-2.0*i)/100.0)
        # print("bufo_size", bufo_size, "bufw_size", bufw_size, "bufi_size", bufi_size)
        # both cases are possible;
        res = opti_deconv_buffer(subs)

        if best_res is None:
            best_res = list(res[0:4])
        elif res[4] and best_res[1] > res[1]:
                best_res = list(res[0:4])
        elif res[4] and best_res[1] == res[1] and best_res[0] > res[0]:
                best_res = list(res[0:4])

        # print(res)

    print("[Best]", best_res)
    return best_res


# optimize one layer
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

