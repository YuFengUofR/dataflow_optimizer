#!/usr/bin/python2.7
import numpy as np

# NOTE: Naming convention #
# all the attributes should defined as lower cases concat with ``_''
# all the function should be defined as lower cases with upper case initial,
# started with lower case;
# all the function should be define as lower cases with upper case initial,
# started with upper case;

# the notation in systolic array in (y, x) !!
#     ---> x 
#     ___________________
#  | |(0,0)  ...  (0,n-1)|
#  | |  .                |
#  v |  .   .            |
#  y |  .     .          |
#    |          .        |
#    |                   |
#    |(n-1,0)...(n-1,n-1)|
#    |___________________|



class MacUnit:
    def __init__(self):
        self.weight = 0
        self.input = 0
        self.accumulated_val = 0

    def mac(self):
        self.accumulated_val += self.weight*self.input

    def flash(self):
        ret = self.accumulated_val
        self.weight = 0
        self.input = 0
        self.accumulated_val = 0
        return ret

    def val(self):
        return self.accumulated_val

class SystolicArray:
    # array is design as row-major index dictionary
    # array[i, j] stands for ith row and jth column;
    def __init__(self, size, mc):
        # connect to a memory controller
        self.mc = mc
        # define the size of systolic array
        self.size = size
        # define the input port
        self.row_i = np.zeros(size)
        # define the weight port
        self.col_w = np.zeros(size)
        # define systolic array
        self.array = {}
        for y in range(size):
            for x in range(size):
                self.array[y, x] = MacUnit();

    # major operation for MAC operation
    def operation(self):
        # feed the data into feeding buffer;
        self.feedInput()
        self.feedWeight()
        # shift the data into systolic array;
        self.shiftData()
        # do the computation
        self.macOperation()

    # feed the data from the memory controller
    def feedInput(self):
        # get input from memory controller;
        in_arr = self.mc.getInput()
        for x in range(self.size):
            self.row_i[x] = in_arr[x]

    def feedWeight(self):
        # get weight from the memory controller
        w_arr = self.mc.getWeights()
        for y in range(self.size):
            self.col_w[y] = w_arr[y]

    # simulate the shift data in the systolic array
    def shiftData(self):
        self.shiftWeight()
        self.shiftInput()

    def shiftWeight(self):
        # reversely shift the data
        for x in range(self.size-1, 0, -1):     # x -- row index
            for y in range(self.size):          # y -- col index
                self.array[y, x].weight = self.array[y, x-1].weight

        # last, feed the 0th col with weight 
        for y in range(self.size):
            self.array[y, 0].weight = self.col_w[y]


    def shiftInput(self):
        # reversely shift the data
        for y in range(self.size-1, 0, -1):     # y -- col index
            for x in range(self.size):          # x -- row index
                self.array[y, x].input = self.array[y-1, x].input

        # last, feeed the 0th row with input
        for x in range(self.size):
            self.array[0, x].input = self.row_i[x]

    def macOperation(self):
        for i in range(self.size):
            for j in range(self.size):
                self.array[i,j].mac()

    def flashData(self):
        ret_mat = outputData()
        # send data to the memory controller
        self.mc.send_data(ret_mat)
        return ret_mat

    def outputData(self):
        # init the matrix
        ret_mat = np.zeros((self.size, self.size))
        # copy the result
        for i in range(self.size):
            for j in range(self.size):
                ret_mat[i, j] = self.array[i,j].val()

        return ret_mat

    def outputWeights(self):
        # init the matrix
        ret_mat = np.zeros((self.size, self.size))
        # copy the result
        for i in range(self.size):
            for j in range(self.size):
                ret_mat[i, j] = self.array[i,j].weight

        return ret_mat

    def outputInput(self):
        # init the matrix
        ret_mat = np.zeros((self.size, self.size))
        # copy the result
        for i in range(self.size):
            for j in range(self.size):
                ret_mat[i, j] = self.array[i,j].input

        return ret_mat

