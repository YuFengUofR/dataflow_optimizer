#!/usr/bin/python2.7
import numpy as np

class MemoryController:
    """docstring for MemoryController"""
    def __init__(self, arr_size, tile_x, tile_y, w_size):
        self.arr_size = arr_size
        # define the tile sizes for input and weights
        self.in_tile = {"x": tile_x, "y": tile_y}
        # because the weight size is already known, compute the row number of weights
        self.w_tile = arr_size/w_size
        self.w_size = w_size
        self.read_counter = {}
        # record the read_counter in x and y
        self.read_counter["x"] = 0 
        self.read_counter["y"] = 0
        self.read_cnt = {"in": [], "w":[]}
        # self.read_cnt is the individual systolic array feeding counter
        for i in range(arr_size):
            self.read_cnt["in"].append(-i)
            self.read_cnt["w"].append(-i)

    def readInput(self, width, height, channel):
        # TODO: need to read config file
        h = height
        w = width
        c = channel
        self.input = np.ones((w, h, c))
        self.h = h
        self.w = w
        self.c = c

    def readWeights(self, width, height, channel, n_weights):
        # TODO: need to read config file
        h = height
        w = width
        c = channel
        n = n_weights
        self.input = np.ones((w, h, c, n))
        sef.n_weights

    # this function only initial x axis
    # in order to record the offset of each systolic array
    def init_read_counter():
        size = self.arr_size
        self.read_cnt["in"].clear()
        self.read_cnt["w"].clear()
        # self.read_cnt is the individual systolic array feeding counter
        for i in range(size):
            self.read_cnt["in"].append(-i)
            self.read_cnt["w"].append(-i)

    # given the systolic array size, send a amount of data;
    def getInput(self):
        ret_arr = []
        x = self.read_counter["x"]
        y = self.read_counter["y"]
        # copy the date from the input buffer
        # store in a row-major fashion
        for i in range(self.in_tile["x"]):         # x
            for j in range(self.in_tile["y"]):     # y
                k = i*self.in_tile["y"]+j          # z

                if self.read_cnt["in"][k] >= 0 and \
                            self.read_cnt["in"][k] < self.c:
                    ret_arr.append(self.input[i][j][k])
                else:
                    ret_arr.append(0)

                self.read_cnt["in"][k] += 1

        return ret_arr

    # given the systolic array size, send a amount of data;
    def getWeights(self):
        # TODO: ret ones at this time now;
        ret_arr = []
        x = self.read_counter["x"]
        y = self.read_counter["y"]
        # copy the date from the input buffer
        # store in a row-major fashion
        for i in range(self.in_tile["x"]):         # x
            for j in range(self.in_tile["y"]):     # y
                k = i*self.in_tile["y"]+j          # z

                if self.read_cnt["in"][k] >= 0 and \
                            self.read_cnt["w"][k] < self.c:
                    ret_arr.append(self.input[i][j][k])
                else:
                    ret_arr.append(0)

                self.read_cnt["w"][k] += 1

        return ret_arr



        









