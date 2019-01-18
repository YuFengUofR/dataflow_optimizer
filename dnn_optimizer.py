from layer_optimizer import optimize, setup_hardware

# a list to store the dnn configuration 
dnn = []

# a list to store all the optimization results
results = []

# import dnn network descrtiption into the system;
def import_dnn(filename=None):
    # clear all the previous contents;
    del dnn[:]
    # the format is 
    # (width, height, in_channel, out_channel,
    #  kenrel_width, kernel_height, stride, Deconv?)    
    dnn.append([512, 384, 6, 64, 7, 7, 2, False])
    dnn.append([256, 192, 64, 128, 5, 5, 2, False])
    dnn.append([128, 96, 128, 256, 5, 5, 2, False])
    dnn.append([64, 48, 256, 256, 3, 3, 1, False])
    dnn.append([64, 48, 256, 512, 3, 3, 1, False])
    dnn.append([32, 24, 512, 512, 3, 3, 1, False])
    dnn.append([32, 24, 512, 512, 3, 3, 2, False])
    dnn.append([16, 12, 512, 512, 3, 3, 1, False])
    dnn.append([16, 12, 512, 1024, 3, 3, 2, False])
    dnn.append([8, 6, 1024, 512, 5, 5, 2, True])
    dnn.append([16, 12, 512, 256, 5, 5, 2, True])
    dnn.append([32, 24, 256, 128, 5, 5, 2, True])
    dnn.append([64, 48, 128, 64, 5, 5, 2, True])


# The hardware constraints are:
#   1. the on-chip buffer size; 
#   2. the memory bandwidth; (Unit in bytes/cycle) 
#   3. the systolic array size;
def hardware_constraints():
    systolic_arr_size = 16.0;
    memory_bandwidth = 1.0;
    buffer_size = 0.25*1024*1024;
    return [systolic_arr_size, memory_bandwidth, buffer_size]

# the main routine of optimizing the dnn.
def opti_dnn():
    global results
    # clear the result first
    del results[:]

    # optimize for each layer
    for layer in dnn:
        print("[Layer]",layer)
        if layer[-1] == True:
            # if the convolution size is odd;
            if layer[5]%2 == 1:
                sub1 = layer
                sub1[4] = (sub1[4]+1)/2
                sub1[5] = (sub1[5]+1)/2
                results.append(optimize(sub1))
                sub2 = layer
                sub2[4] = (sub2[4]-1)/2
                sub2[5] = (sub2[5]-1)/2
                results.append(optimize(sub2))
            # if the convolution size is even;
            else:
                sub = layer
                sub[4] = sub[4]/2
                sub[5] = sub[5]/2
                results.append(optimize(sub))
        else:
            # scale down the ifmap to the ifmap based on the stride size.
            layer[0] = layer[0]/layer[-2]
            layer[1] = layer[1]/layer[-2]
            results.append(optimize(layer))

    for res in results:
        print(res)

if __name__== '__main__':
    # import the dnn
    import_dnn()
    # set up the hardware configuration
    setup_hardware(hardware_constraints())
    # start the optimization main routine
    opti_dnn()


