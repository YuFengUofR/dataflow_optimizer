#!/usr/bin/python2.7

# public library
import math
import numpy as np

# my own module
from layer_static_method import *

###############################################################
#                       general process                       #
###############################################################
class LayerExhaustiveSearcher(LayerStaticMethod):

    # array to store the result from the four different results
    res = []

    """docstring for LayerExhaustiveSearcher"""
    def __init__(self, data, sys_info):
        super(LayerExhaustiveSearcher, self).__init__(data, sys_info)
        self.rets = []

    # optimize one layer
    def optimize(self):
        self.res = []
        layer_info = self.data
        # set up the new layer information
        [self.W, self.H, self.Ci] = layer_info["ifmap"]
        self.Co = layer_info["out_channel"]
        [self.K_w, self.K_h] = layer_info["kernel"]
        self.S = layer_info["stride"]

        # print("##[LAYER]##", self.W, self.H, self.Ci, self.Co, self.K_w, self.K_h)

        for i in range(1, 20):
            self.bufi_size = self.buf_size*i/20.0
            for j in range(1, 20):
                self.bufw_size = self.buf_size*j/20.0

                self.res = []
                # if sum of bufi and bufw is over the self.buf_size
                # we should skip it.
                if (self.bufi_size + self.bufw_size) > self.buf_size:
                    continue

                self.bufo_size = self.buf_size - self.bufi_size - self.bufw_size
                # both cases are possible;
                self.opti_buffer()

                if len(self.res) == 0:
                    continue

                # choose the larger value as the bottleneck
                row_major_res = None
                if (self.res[0]["total_cycle"] < self.res[1]["total_cycle"]):
                    row_major_res = self.res[1]
                else:
                    row_major_res = self.res[0]

                # choose the larger value as the bottleneck
                channel_major_res = None
                if (self.res[2]["total_cycle"] < self.res[3]["total_cycle"]):
                    channel_major_res = self.res[3]
                else:
                    channel_major_res = self.res[2]

                # return the shortest value as the perferred compute ordering.
                ret = None
                if (row_major_res["total_cycle"] < channel_major_res["total_cycle"]):
                    ret = dict(row_major_res)
                else:
                    ret = dict(channel_major_res)

                self.rets.append(ret)

        ret  = dict(self.rets[0])

        for item in self.rets:
            if ret["total_cycle"] > item["total_cycle"]:
                ret = dict(item)
            if ret["total_cycle"] == item["total_cycle"] and \
                ret["total_transfer"] > item["total_transfer"]:
                ret = dict(item)

        return ret
