#!/usr/bin/python2.7
import numpy as np

# buf info:
# 
# the buf bank has a same size buf as systolic array:
# (size, size) with additional data buf to store some 
# extra data that can't immediately consumed;
# 
#  Q:How to store the data?
#  A: for any systolic array, only last row and last colume will 
#     have the data that potentially need to be store in the bank 
#     for the next computation
# 
#                    last column 
#   -------->  x      |
#  |  ________________V__
#  | |(0,0)  ...  (0,n-1)|
#  | |  .                |  
#  v |  .   .            |   
#  y |  .     .          |
#    |          .        |
#    |                   |
#    |(n-1,0)...(n-1,n-1)|   <- last row
#    |___________________|
#               
#               
# for a 16x16 tile

# This class is to mimic the onship buffer storage

class OnchipBuffer:
    """docstring for buf"""
    def __init__(self, sys_arr, size, tile_x, tile_y, w_size, block_x):
        # set on-chip buffer as 0
        self.on_chip_store = 0
        # set the sys_arr reference 
        self.sys_arr = sys_arr
        # set the size of sys_arr
        self.size = size
        # set the inital buf for systolic array
        self.buf = np.zeros((size, size))
        # set the block size so we can calcule the actual onchip storage status
        self.block_x = block_x
        # record the tile (w, h)
        self.tile_x = tile_x
        self.tile_y = tile_y
        self.w_size = w_size
        # num of weight in sys_arr
        num = size / (w_size*w_size)
        # set additional buf for the paritial sum
        self.extra_x = np.zeros((tile_x*2*num, 2))
        self.extra_y = np.zeros((tile_y*2*num, 2))


    # if each weight (4x4) can be feed into systolic array,
    # then, every single time, only need to store additional 
    # data is (tile_x+tile_y)*2 for the tile on the left or below; 
    #
    # Q: Why I only store the partial in y axis
    # A: because the partial sum in the last x-axis need to be 
    #    consumed until the next line be computed;
    #    If we store them, we need line_length*3*4-bytes;
    #    Not that much worthy!
    def consumeData(self, index):
        # flash out the data from the systolic array
        self.buf = self.sys_arr.outputData()
        # compute the output
        # self.compute4TypeOutput()
        write_out = (self.tile_x*2)*(self.tile_y*2)
        if self.on_chip_store != 0 and index % self.block_x != 0:
            self.on_chip_store -= self.tile_y*4
        if index / self.block_x >= 1:
            self.on_chip_store -= self.tile_x*4

        self.on_chip_store += (self.tile_y+self.tile_x)*2*2
        print("[on-chip]== store %d 4-byptes PARTIAL SUM on CHIP ==" % self.on_chip_store)
        print("[Write]== store %d 4-bytes OUTPUT into MEM ==" % write_out)

        return write_out

    # this is the function to compute 4 different type of DeConv values
    def compute4TypeOutput(self):
        # some initials
        size_c = self.size/self.w_size/self.w_size
        # construct a return matirx
        ret_arr = np.zeros((self.tile_x*2, self.tile_y*2, size_c))
        # make some reference
        tile_y = self.tile_y
        tile_x = self.tile_x
        buf = self.buf
        # compute the number of weights computed in the buffer
        for k in range(size_c):
            offset = k*(self.w_size*self.w_size)
            for j in range(tile_y-1):
                for i in range(tile_x-1):
                    # compute the index of 1st compute element
                    idx = i+j*tile_x
                    # compute for total result from the partial sum (4 types)
                    # the following computation assuming weights are 4x4
                    val1 = buf[idx, 5] + buf[idx+1, 7] \
                        + buf[idx+tile_x, 13] + buf[idx+tile_x+1, 15]
                    val2 = buf[idx, 4] + buf[idx+1, 6] \
                        + buf[idx+tile_x, 12] + buf[idx+tile_x+1, 14]
                    val3 = buf[idx, 1] + buf[idx+1, 3] \
                        + buf[idx+tile_x, 9] + buf[idx+tile_x+1, 11]
                    val4 = buf[idx, 0] + buf[idx+1, 2] \
                        + buf[idx+tile_x, 8] + buf[idx+tile_x+1, 10]
                    # write to the writeback buffer
                    ret_arr[2*i, 2*j, k] = val1
                    ret_arr[2*i+1, 2*j, k] = val2
                    ret_arr[2*i, 2*j+1, k] = val3
                    ret_arr[2*i+1, 2*j+1, k] = val4

        # compute the partial sum in last col
        # compute the number of weights computed in the buffer
        for k in range(size_c):
            offset = k*(self.w_size*self.w_size)
            for j in range(tile_y-1):
                # compute the index of 1st compute element
                idx = (1+j)*tile_x-1
                # compute for total result from the partial sum (4 types)
                # the following computation assuming weights are 4x4
                val1 = buf[idx, 5] + buf[idx+tile_x, 13]
                val2 = buf[idx, 4] + buf[idx+tile_x, 12]
                val3 = buf[idx, 1] + buf[idx+tile_x, 9]
                val4 = buf[idx, 0] + buf[idx+tile_x, 8]
                # write to the writeback buffer
                self.extra_y[2*j+offset, 0] = val1
                self.extra_y[2*j+offset, 1] = val2
                self.extra_y[2*j+1+offset, 0] = val3
                self.extra_y[2*j+1+offset, 1] = val4

        # compute the partial sum in last row
        for k in range(size_c):
            offset = k*(self.w_size*self.w_size)
            for j in range(tile_y-1):
                # compute the index of 1st compute element
                idx = (1+j)*tile_x-1
                # compute for total result from the partial sum (4 types)
                # the following computation assuming weights are 4x4
                val1 = buf[idx, 5] + buf[idx+tile_x, 7]
                val2 = buf[idx, 4] + buf[idx+tile_x, 6]
                val3 = buf[idx, 1] + buf[idx+tile_x, 3]
                val4 = buf[idx, 0] + buf[idx+tile_x, 2]
                # write to the writeback buffer
                self.extra_x[2*j+offset, 0] = val1
                self.extra_x[2*j+offset, 1] = val2
                self.extra_x[2*j+1+offset, 0] = val3
                self.extra_x[2*j+1+offset, 1] = val4

        

