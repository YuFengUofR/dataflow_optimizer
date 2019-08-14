#!/usr/bin/python2.7

# public library
import math
import numpy as np

# own library
from layer_base_method import *

class Layer3dbaseMethod(LayerBaseMethod):
    """docstring for Layer3dBaseMethod"""
    # info for systolic array
    A = None      # systolic array dimension

    # memory bandwith number of bytes can be transferred.
    B = None

    # on-chip buffer size
    buffer_size = None
    # info for weights
    K_w = None       # kernel width
    K_h = None       # kernel height
    k_d = None       # kernel dispairty
    S = None         # stride size

    # input layer dimension
    H = None        # height of ofmap
    W = None        # width of ofmap
    D = None        # disparity of ofmap
    Ci = None       # channels for weights
    Co = None       # channels for ofmap

    # on-chip buffer size
    bufi_size = None
    bufo_size = None
    bufw_size = None

    # array to store the result from the four different results
    res = []

    def __init__(self, data, sys_info):
      super(Layer3dBaseMethod, self).__init__(data, sys_info)

    def init_setup(self):
        self.res = []
        layer_info = self.data
        # set up the new layer information
        [self.W, self.H, self.D, self.Ci] = layer_info["ifmap"]
        self.Co = layer_info["out_channel"]
        [self.K_w, self.K_h, self.K_d] = layer_info["kernel"]
        self.S = layer_info["stride"]

    # variables for optimization
    # this two has been encodes as x[3] = {c_0, h_0, w_0, d_0};
    # c_0  # number of channels per batch;
    # h_0, w_0, d_0 # the dimensions of tile per batch;

    ###############################################################
    #                       general process                       #
    ###############################################################

    def row_major_data_transfer(h_0, w_0, d_0, c_0):
        # ofmap, ifmap and kernel tile size
        S_2 = (self.K_h+1) / 2
        ofmap_tile_size = h_0*w_0*d_0*c_0
        ifmap_tile_size = ((self.S*h_0+S_2) * (self.S*w_0+S_2) * \
                           (self.S*d_0+S_2) * self.Ci)
        kernel_tile_size = self.K_h*self.K_w*self.K_d*self.Ci*c_0

        # calculate the total batch
        total_batch = (self.H*self.W*self.D*self.Co) / ofmap_tile_size

        # ofmap + ifmap transfer
        total_transfer = (ofmap_tile_size + ifmap_tile_size) * \
            (total_batch - self.Co/c_0)

        # add additional data transfer
        total_transfer += (ofmap_tile_size + kernel_tile_size) * (self.Co/c_0)

        return total_transfer

    def channel_major_data_transfer(h_0, w_0, d_0, c_0):
        S_2 = (self.K_h+1) / 2

        # ofmap and ifmap tile size
        ofmap_tile_size = h_0*w_0*d_0*c_0
        ifmap_tile_size = ((self.S*h_0+S_2) * (self.S*w_0+S_2) * \
                           (self.S*d_0+S_2) * self.Ci)
        kernel_tile_size = self.K_h*self.K_w*self.K_d*self.Ci*c_0

        # calculate the total batch
        total_batch = (self.H*self.W*self.D*self.Co) / ofmap_tile_size

        # ofmap + weight transfer
        total_transfer = (ofmap_tile_size + kernel_tile_size) * \
            (total_batch - (self.H*self.W*self.D)/(h_0*w_0*d_0))

        # add additional data transfer
        total_transfer += (ofmap_tile_size + ifmap_tile_size) \
            * (self.H*self.W*self.D)/(h_0*w_0*d_0)

        return total_transfer

    def systolic_array_utilization(self, xi, area):
        area_size = area[0] * area[1] *area[2]
        A = self.A
        total_usage = xi * area_size
        round_up_val = (math.ceil(float(xi/A))*A) \
                        * (math.ceil(float(area_size)/A)*A)

        return total_usage / round_up_val

    def compute_bound_cycle(self, util_rate):
        # total number of ops
        total_computation = (self.H*self.W*self.Dself.Co)*\
            (self.Ci*self.K_h*self.K_w*self.K_d)

        # systolic array calculation capacity
        comp_cap = (self.A*self.A) * util_rate

        return total_computation / comp_cap

    def process_parameter(self, x, row_major, comp_bound):
        bound = "C"
        # make the tile size even for every batch
        c_0 = min(self.Co/math.ceil(self.Co/round(x[0])), self.Co)
        w_0 = min(self.W/math.ceil(self.W/round(x[1])), self.W)
        h_0 = min(self.H/math.ceil(self.H/round(x[2])), self.H)
        d_0 = min(self.D/math.ceil(self.D/round(x[3])), self.D)

        #compute the total number of elements needed to be updated
        # if it is row-major.
        if row_major:
            # (ofmap + ifmap)*total_batch + (ofmap+weights)*Co/c_0
            total_transfer = self.row_major_data_transfer(h_0, w_0, d_0, c_0)

        # compute the total number of elements needed to be updated
        # if it is channel-major.
        else:
            # (ofmap + weights)*total_batch + (ofmap+ifmap)*(H*W)/(h_0*w_0)
            total_transfer = self.channel_major_data_transfer(h_0, w_0, d_0, c_0)

        # compute the utilization of systolic array
        util_sys_arr = self..systolic_array_utilization(c_0, [w_0, h_0, d_0])

        # compute the utilization of systolic array
        util_buf = buffer_constraint1([c_0, w_0, h_0, d_0])/buffer_size

        if util_buf > 1.01:
            return
        # calculate the amount of cycles of computing all elements.
        if comp_bound:
            bound = "C"
            total_cycle = self.compute_bound_cycle(util_sys_arr)
        else:
            bound = "M"
            total_cycle = total_transfer/self.B

        ret = {
            "total_transfer": round(total_transfer),
            "total_cycle": round(total_cycle),
            "systolic_array_utilization": util_sys_arr,
            "buffer_utilization": util_buf,
            "c_0, w_0, h_0, d_0": [c_0, w_0, h_0, d_0],
            "Tile size" : [self.Co/c_0, self.W/w_0, self.H/h_0, self.D/d_0],
            "Bound" : bound
        }
        self.res.append(ret)
        return

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
    # [h_0*w_0*d_0*c_0+(S*h_0+2)(S*w_0+2)(S*d_0+2)*Ci]
    # *(H*W*D*Co)/(h_0*w_0*d_0*c_0)
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


