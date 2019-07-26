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
        super(LayerExhaustiveSearcher, self).__init__(data, sys_info, None)
        self.rets = []

    # optimize one layer
    def optimize(self):
        layer_info = self.data
        # set up the new layer information
        [self.W, self.H, self.Ci] = layer_info["ifmap"]
        self.Co = layer_info["out_channel"]
        [self.K_w, self.K_h] = layer_info["kernel"]
        self.S = layer_info["stride"]

        for i in range(1, 20):
            self.bufi_size = self.buf_size*i/20.0
            for j in range(1, 20):
                self.bufw_size = self.buf_size*j/20.0
                # optimize one buffer partition
                self.res = []
                self.optimize_one_buffer_partition()

        ret  = dict(self.rets[0])

        for item in self.rets:
            if ret["total_cycle"] > item["total_cycle"]:
                ret = dict(item)
            if ret["total_cycle"] == item["total_cycle"] and \
                ret["total_transfer"] > item["total_transfer"]:
                ret = dict(item)

        return ret
