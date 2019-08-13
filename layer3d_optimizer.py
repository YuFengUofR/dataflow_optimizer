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

class Layer3dOptimizer(Layer3dBaseMethod, LayerOptimizer):
    """docstring for Layer3dOptimizer"""
    def __init__(self, data, sys_info, combined=False):
        super(Layer3dOptimizer, self).__init__(data, sys_info)

    # variables for optimization
    # this two has been encodes as x[3] = {c_0, h_0, w_0, d_0};
    # c_0  # number of channels per batch;
    # h_0, w_0, d_0 # the dimensions of tile per batch;

    ###############################################################
    #                       general process                       #
    ###############################################################

    def process_parameter(self, x, row_major, comp_bound):
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


    def init_guess(self):
        x0 = [min(self.A, self.Co), \
              min(math.floor(math.sqrt(self.A)), self.H), \
              min(math.floor(math.sqrt(self.A)), self.W), 1]
        return x0

    def veriable_boundary(self):
        return ((min(self.A, self.Co), self.Co),
                (min(math.floor(math.sqrt(self.A)), self.H), self.H), \
                (min(math.floor(math.sqrt(self.A)), self.W), self.W), \
                (1, self.D))

    ###############################################################
    #                     general computations                    #
    ###############################################################
    def ofmap_tile(self, x):
        return x[0]*x[1]*x[2]*x[3]

    def weight_tile(self, num):
        return self.Ci*self.K_h*K_w*self.K_d*num

    def ifmap_tile(self, x):
        S_2 = (self.K_h+1) / 2
        return self.Ci*(self.S*x[1]+S_2)*(self.S*x[2]+S_2)*(self.S*x[3]+S_2)

    def total_ofmap_size(self):
        return self.H*self.W*self.D*self.Co

    def total_weight_size(self):
        return self.weight_tile(self.Co)


    ###############################################################
    #                     general constraints                     #
    ###############################################################
    # the low bound of buffer size;
    # make sure the buffer utilization is always larger than 0
    def buffer_constraint1(self, x):
        return LayerOptimizer.buffer_constaint1(self, x)

    # the upper bound of the buffer size;
    # make sure the buffer utilization is
    # always smaller than buffer size;
    def buffer_constraint2(self, x):
        return self.buf_size - self.buffer_constraint1(x)

    ###############################################################
    #       row-major constraint solving obj and constraints      #
    ###############################################################

    # the minimization objective of row-major
    # this objective is a simplified expression of
    # [h_0*w_0*d_0*c_0+(S*h_0+2)(S*w_0+2)(S*d_0+2)*Ci]*(H*W*D*Co)/(h_0*w_0*d_0*c_0)
    # + [K^3*Ci+h_0*w_0*d_0*c_0]*Co/c_0
    def row_major_mem_obj(self, x):
      return (self.ofmap_tile(x) + self.ifmap_tile(x)) \
          * (self.total_ofmap_size()/self.ofmap_tile(x) - self.Co/x[0]) \
          + self.total_weight_size()/x[0] + x[1]*x[2]*x[3]*self.Co

    def row_major_comp_obj(self, x):
        return self.total_ofmap_size() / self.ofmap_tile(x)

    # make sure the load for row-major is always less than
    # load for channel-major, range : [0, +inf]
    def row_major_constraint(self, x):
        # simplified from K^3*Ci*c_0 > C*(S^3*h_0*w_0*d_0)
        return self.K_h*self.K_w*self.K_d*x[0] - \
            (self.S*x[1]+S_2)*(self.S*x[2]+S_2)*(self.S*x[3]+S_2);

    # make sure the process is always memory-bound;
    # which is the latency for memory access is always
    # greater than lantecy of compute;
    # (c_0*(h_0*w_0*d_0)+C*((S*h_0+2)*(S*w_0+2)*(S*d_0+2))/B
    # >= (K^3*Ci/A^2)*c_0*w_0*d_0*h_0
    # range : [0, +inf]
    def row_major_mem_bound_constraint(self, x):
      return (self.ofmap_tile(x) + self.ifmap_tile(x)) / self.B \
          - self.weight_tile(1)/(self.A*self.A)*self.ofmap_tile(x))

    # make sure the process is always compute-bound;
    # which is the latency for compute is always
    # greater than lantecy of memory access;
    # (c_0*(h_0*w_0*d_0)+Ci*((S*h_0+2)*(S*w_0+2)*(S*d_0+2))/B
    # <= (K^3*Ci/A^2)*c_0*w_0*h_0*d_0
    # range : [0, +inf]
    def row_major_comp_bound_constraint(self, x):
        return self.weight_tile(1) / (self.A*self.A)*self.ofmap_tile(x) \
            - (self.ofmap_tile(x) + self.ifmap+_tile(x)) / self.B

    ###############################################################
    #     channel-major constraint solving obj and constraints    #
    ###############################################################

    # the minimization objective of channel-major
    # this is the simplified expression of
    # (K^3*Ci*c_0+h_0*w_0*d_0*c_0)*(H*W*D*Co)/(h_0*w_0*d_0*c_0)
    # + [(S*h_0+2)(S*w_0+2)(S*d_0+2)*Ci + h_0*w_0*d_0*c_0]*(H*W*D)/(h_0*w_0*d_0)
    def channel_major_mem_obj(self, x):
        return (self.total_weight_size)/(x[1]*x[2]*x[3]) + \
                (self.S*x[1]+S_2)*(self.S*x[2]+S_2)*(self.S*x[3]+S_2)/\
                (x[1]*x[2]*x[3])

    def channel_major_comp_obj(self, x):
        return self.total_ofmap_size()/(x[1]*x[2]*x[0]*x[3])

    # make sure the load for channel-major is always less than
    # load for row-major, range : [0, +inf]
    def channel_major_constraint(self, x):
        S_2 = (self.K_h+1)/2
        # simplified from K^3*Ci*c_0 <= Ci*((S*h_0+2)*(S*w_0+2))
        return (self.S*x[1]+S_2)*(self.S*x[2]+S_2)*(self.S*x[3]+S_2) \
            - self.K_h*self.K_w*self.K_d*x[0];

    # make sure the process is always memory-bound;
    # which is the latency for memory access is always
    # greater than lantecy of compute;
    # c_0*(h_0*w_0+K^3*C)/B >= (K^3*C/A^2)*c_0*(h_0*w_0)
    # range : [0, +inf]
    def channel_major_mem_bound_constraint(self, x):
        return (x[1]*x[2]*x[3]+self.weight_tile(1)) / self.B \
            - self.weight_tile(1)/(self.A*self.A)*x[1]*x[2]*x[3]


    # make sure the process is always memory-bound;
    # which is the latency for memory access is always
    # greater than lantecy of compute;
    # c_0*(h_0*w_0+K^3*C)/B >= (K^3*C/A^2)*c_0*(h_0*w_0*d_0)
    # range : [0, +inf]
    def channel_major_comp_bound_constraint(x):
        return self.K_h*self.K_w*self.K_d*Co/(self.A*self.A)*x[1]*x[2]*x[3] \
            - (x[1]*x[2]+self.K_h*self.K_w*self.K_d*self.Co)/self.B


