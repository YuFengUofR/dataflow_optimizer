#!/usr/bin/python2.7
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import FuncFormatter
import numpy as np

# This is a function to plot the utilization of buffer 
# and systolic array for a single DNN under one particular
# hardware configuration
def plot_util_buf():
    # initialize the array from the result first.
    buf_util = [[0.24277782440185547, 0.145782470703125, 0.145782470703125, 0.112823486328125, 0.10003662109375, 0.08685302734375, 0.1278076171875, 0.17706298828125, 0.189422607421875, 0.189422607421875, 0.1187744140625, 0.16717529296875, 0.1292724609375, 0.236083984375, 0.16717529296875, 0.17706298828125, 0.1187744140625, 0.16717529296875, 0.1292724609375, 0.236083984375, 0.16717529296875, 0.17706298828125, 0.1187744140625, 0.16717529296875, 0.1292724609375, 0.236083984375, 0.16717529296875, 0.17706298828125],
                [0.24277782440185547, 0.145782470703125, 0.145782470703125, 0.112823486328125, 0.10003662109375, 0.08685302734375, 0.1278076171875, 0.17706298828125, 0.189422607421875, 0.189422607421875, 0.1187744140625, 0.16717529296875, 0.1292724609375, 0.236083984375, 0.236083984375, 0.13751220703125, 0.1187744140625, 0.16717529296875, 0.1292724609375, 0.236083984375, 0.236083984375, 0.13751220703125, 0.1187744140625, 0.16717529296875, 0.1292724609375, 0.236083984375, 0.236083984375, 0.13751220703125],
                [0.00023555755615234375, 0.712188720703125, 0.712188720703125, 0.803497314453125, 0.02789306640625, 0.42120361328125, 0.02880859375, 0.01842407582452624, 0.01287841796875, 0.01287841796875, 0.39569091796875, 0.6201171875, 0.7254638671875, 0.48388671875, 0.461181640625, 0.898681640625, 0.39569091796875, 0.6201171875, 0.7254638671875, 0.48388671875, 0.461181640625, 0.898681640625, 0.39569091796875, 0.6201171875, 0.7254638671875, 0.48388671875, 0.461181640625, 0.898681640625],
                [0.00023555755615234375, 0.712188720703125, 0.712188720703125, 0.803497314453125, 0.02789306640625, 0.42120361328125, 0.02880859375, 0.01842407582452624, 0.01287841796875, 0.01287841796875, 0.39569091796875, 0.6201171875, 0.7254638671875, 0.48388671875, 0.48388671875, 0.016631155303030304, 0.39569091796875, 0.6201171875, 0.7254638671875, 0.48388671875, 0.48388671875, 0.016631155303030304, 0.39569091796875, 0.6201171875, 0.7254638671875, 0.48388671875, 0.48388671875, 0.016631155303030304]]

    x_axis_ls = range(1, len(buf_util[0])+1)
    plt.rc('font', size=10)
    ax = plt.figure(figsize=(8, 3)).add_subplot(111)
    ax.set_ylabel('Buf util', fontsize=14, fontweight='bold')
    plt.xticks(rotation=60)
    plt.setp(ax.spines.values(), linewidth=2)

    p = [None, None, None, None]
    colors = ['#71985E', '#5b87f2', '#FFBF56', '#8154D1']
    markers = ['o', 'v', 'd', '^']
    for i in range(len(buf_util)):
        p[i] = ax.plot(x_axis_ls, buf_util[i], color=colors[i], linestyle=':', linewidth=2, \
                marker=markers[i],markersize=8, markeredgewidth=1.5, markeredgecolor='k');

    plt.xticks(x_axis_ls, [ "Layer" + str(n) for n in x_axis_ls])
    ax.set_ylim(0.0, 1.0)
    ax.tick_params(axis="y",direction="in")
    ax.tick_params(axis="x",direction="in")
    plt.subplots_adjust(left=0.1, bottom=0.25, right=0.9, top=0.9,
                    wspace=0.2, hspace=0.2)
    plt.grid(color='grey', which='major', axis='y', linestyle='--')
    plt.legend((p[0][0], p[1][0], p[2][0], p[3][0]), ('Static','Split','Dynamic','Combine'), \
            bbox_to_anchor=(0., 1.01, 1., .101), loc=3,
            ncol=4, mode="expand", borderaxespad=0., frameon=False)
    ax.set_axisbelow(True)
    
    plt.savefig("layer_buf_util.pdf");


def profile_layer_cycle():
    # initialize the array from the result first.
    total_cycle = [[3421332.0, 4668864.0, 4668864.0, 3493824.0, 2406912.0, 1826304.0, 2233027.9716784675, 639016882.5708885, 318992417.8512397, 318992417.8512397, 340182528.0, 239631330.96408316, 165807360.0, 59719680.00000001, 239631330.96408316, 639016882.5708885, 340182528.0, 239631330.96408316, 165807360.0, 59719680.00000001, 239631330.96408316, 639016882.5708885, 340182528.0, 239631330.96408316, 165807360.0, 59719680.00000001, 239631330.96408316, 639016882.5708885],
            [3421332.0, 4668864.0, 4668864.0, 3493824.0, 2406912.0, 1826304.0, 2233027.9716784675, 639016882.5708885, 318992417.8512397, 318992417.8512397, 340182528.0, 239631330.96408316, 165807360.0, 59719680.00000001, 59719680.00000001, 159754220.64272213, 340182528.0, 239631330.96408316, 165807360.0, 59719680.00000001, 59719680.00000001, 159754220.64272213, 340182528.0, 239631330.96408316, 165807360.0, 59719680.00000001, 59719680.00000001, 159754220.64272213],
            [311039, 3318609, 3318609, 1244108, 2403840, 1244160, 2408448, 200728320, 89164800, 89164800, 22429301, 44791835, 5612708, 5598720, 5598720, 29867741, 22429301, 44791835, 5612708, 5598720, 5598720, 29867741, 22429301, 44791835, 5612708, 5598720, 5598720, 29867741],
            [311039, 3318609, 3318609, 1244108, 2403840, 1244160, 2408448, 200728320, 89164800, 89164800, 22429301, 44791835, 5612708, 5598720, 5598720, 27899904, 22429301, 44791835, 5612708, 5598720, 5598720, 27899904, 22429301, 44791835, 5612708, 5598720, 5598720, 27899904]]

    x_axis_ls = range(1, len(total_cycle[0])+1)
    plt.rc('font', size=10)
    ax = plt.figure(figsize=(8, 3)).add_subplot(111)
    ax.set_ylabel('Cycle in Logscale', fontsize=14, fontweight='bold')
    plt.xticks(rotation=60)
    plt.setp(ax.spines.values(), linewidth=2)
    p = [None, None, None, None]
    colors = ['#71985E', '#5b87f2', '#FFBF56', '#8154D1']
    markers = ['o', 'v', 'd', '^']

    for i in range(len(total_cycle)):
        p[i] = ax.plot(x_axis_ls, total_cycle[i], color=colors[i], linestyle=':', linewidth=2, \
                marker=markers[i], markersize=8, markeredgewidth=1.5, markeredgecolor='k');
    
    plt.subplots_adjust(left=0.1, bottom=0.25, right=0.9, top=0.9,
                wspace=0.2, hspace=0.2)
    ax.set_ylim(pow(10, 5), pow(10, 11))
    ax.set_yscale('log')
    plt.xticks(x_axis_ls, [ "Layer" + str(n) for n in x_axis_ls])
    ax.tick_params(axis="y",direction="in")
    ax.tick_params(axis="x",direction="in")
    plt.grid(color='grey', which='major', axis='y', linestyle='--')
    plt.legend((p[0][0], p[1][0], p[2][0], p[3][0]), ('Static','Split','Dynamic','Combine'), \
            bbox_to_anchor=(0., 1.01, 1., .101), loc=3,
            ncol=4, mode="expand", borderaxespad=0., frameon=False)
    ax.set_axisbelow(True)
    
    plt.savefig("layer_cycle.pdf");

def profile_speedup():
    # initialize the array from the result first.
    speedup = [[1.23, 2.93, 1.11, 3.19, 1.17, 6.01, 1.45, 5.43, 1.25, 4.5],
                [1.52, 3.93, 1.49, 4.02, 1.47, 8.02, 1.85, 7.43, 1.5, 6.0],
                [1.52, 3.93, 1.49, 4.02, 1.47, 8.02, 1.85, 7.43, 1.5, 6.0]]

    x_axis_ls =[1.1, 1.5, 2.1, 2.5, 3.1, 3.5, 4.1, 4.5, 5.1, 5.5]
    plt.rc('font', size=10)
    ax = plt.figure(figsize=(4, 3)).add_subplot(111)
    ax.set_ylabel('Speedup', fontsize=14, fontweight='bold')
    plt.xticks(rotation=60)
    plt.setp(ax.spines.values(), linewidth=2)
    p = [None, None, None, None]
    colors = ['#FFBF56','#71985E', '#5b87f2', '#8154D1']
    hatches = ['///', '\\\\\\', 'oo', '|||']

    for i in [0,1,2]:
        tmp_x = [x for x in x_axis_ls]
        p[i] = ax.bar(tmp_x, speedup[i], 0.4, align='center', color=colors[i],\
                edgecolor=['k']*len(x_axis_ls), linewidth=1.5, hatch=hatches[i]);
    
    plt.subplots_adjust(left=0.15, bottom=0.25, right=0.9, top=0.8,
                wspace=0.2, hspace=0.2)
    plt.xticks([1.35, 2.35, 3.35, 4.35, 5.35], ['FlowNetS','FlowNetC','GC-Net', 'PSMNet', 'AVG.'])
    ax.tick_params(axis="y",direction="in")
    ax.tick_params(axis="x",direction="in")
    plt.grid(color='grey', which='major', axis='y', linestyle='--')
    plt.legend((p[0][0], p[1][0]), ('Trans.','Reuse'), \
            bbox_to_anchor=(0., 1.01, 1., .101), loc=3,
            ncol=3, mode="expand", borderaxespad=0., frameon=False)
    ax.set_axisbelow(True)
    
    plt.savefig("layer_speedup.pdf");

def profile_energy():
    # initialize the array from the result first.
    energy_ratio = [[0.88, 0.78, 0.93, 0.73, 0.74, 0.71, 0.77, 0.65, 0.8, 0.7],
                [0.78, 0.67, 0.80, 0.48, 0.39, 0.06, 0.36, 0.25, 0.6, 0.5]]

    x_axis_ls =[1.1, 1.5, 2.1, 2.5, 3.1, 3.5, 4.1, 4.5, 5.1, 5.5]
    plt.rc('font', size=10)
    ax = plt.figure(figsize=(4, 3)).add_subplot(111)
    ax.set_ylabel('Energy Saving', fontsize=14, fontweight='bold')
    plt.xticks(rotation=60)
    plt.setp(ax.spines.values(), linewidth=2)
    p = [None, None, None, None]
    colors = ['#FFBF56', '#71985E','#5b87f2', '#8154D1']
    hatches = ['///', '\\\\\\','oo', '|||']

    for i in [0,1]:
        tmp_x = [x for x in x_axis_ls]
        p[i] = ax.bar(tmp_x, energy_ratio[i], 0.4, align='center', color=colors[i],\
                edgecolor=['k']*len(x_axis_ls), linewidth=1.5, hatch=hatches[i]);
    
    plt.subplots_adjust(left=0.15, bottom=0.25, right=0.9, top=0.8,
                wspace=0.2, hspace=0.2)
    plt.xticks([1.3, 2.3, 3.3, 4.3, 5.3], ['FlowNetS','FlowNetC','GC-Net', 'PSMNet', 'AVG.'])
    ax.tick_params(axis="y",direction="in")
    ax.tick_params(axis="x",direction="in")
    ax.set_ylim(0, 1)
    plt.grid(color='grey', which='major', axis='y', linestyle='--')
    plt.legend((p[0][0], p[1][0]), ('Trans.','Reuse'), \
            bbox_to_anchor=(0., 1.01, 1., .101), loc=3,
            ncol=3, mode="expand", borderaxespad=0., frameon=False)
    ax.set_axisbelow(True)
    
    plt.savefig("layer_energy_saving.pdf");


cmap_name="Reds"

def profile_sensitivity_cycle():
    y = ["0.50", "1.00", "1.50", "2.00","2.50", "3.00"]
    x = ["08x08", "16x16", "24x24", "32x32", "40x40", "48x48", "56x56"]

    # raw = np.array([[1346450098.0, 285760083.92, 173093035.84, 72190208.28, 59167128.80, 54689744.96, 43732189.16],
    #                 [1346450098.0, 280878381.52, 169739441.08, 69387075.60, 55496500.68, 46301349.08, 41293463.36],
    #                 [1343452566.4, 276289671.96, 169514604.84, 69116078.36, 55323932.68, 46254284.24, 39475957.12],
    #                 [1343080176.6, 276211914.00, 169428729.36, 69105622.68, 55287091.00, 46179605.24, 39493259.92],
    #                 [1343080176.6, 276154471.28, 169434676.16, 69098476.44, 55267602.12, 46115484.80, 39511993.72],
    #                 [1110879520.4, 276191765.24, 169386728.36, 69065520.36, 55260948.60, 46116038.12, 39507947.84],
    #                 [1104425174.5, 276109688.32, 169334966.44, 69037027.64, 55259580.72, 46077337.36, 39465001.60]])

    # # 1345074228.64, 281512936.40, 173093035.88, 70783676.12, 57778432.0, 48110560.16, 36775251.92],
    # GC-Net
    raw = np.array([[1111124424.92, 276631234.00, 169739441.08, 69386352.08, 55471383.76, 46216657.64, 39723227.12, 29354436.04],
                    [1143452566.24, 276289671.96, 169408406.88, 69116178.44, 55297473.72, 46188296.88, 39475960.52, 29020161.04],
                    [1163109930.28, 276208444.76, 169428729.12, 69105705.96, 55260845.72, 46113307.84, 39493259.00, 29030247.16],
                    [1166334800.08, 276143563.68, 169434674.28, 69098600.52, 55241064.76, 46049312.88, 39511974.24, 29029670.84],
                    [1104479570.28, 276120246.48, 169417224.08, 69065728.00, 55232945.20, 46063724.68, 39479170.52, 29033568.76],
                    [1110877199.84, 276149230.32, 169349295.88, 69065460.64, 55234458.80, 46049820.60, 39507994.52, 29023590.16]])

    # 1345074228.64, 281512936.40, 173093035.88, 70783676.12, 57778432.0, 48110560.16, 36775251.92],
    # Flow-Net-C
    raw = np.array([[49493394, 13266263, 6417728, 3524839, 2407647, 1867746, 1301906, 884975],
                    [48689890, 12363012, 5868066, 3329769, 2307832, 1678266, 1239938, 940589],
                    [48195711, 12250135, 5717865, 3200491, 2233979, 1613569, 1178820, 911407],
                    [48112072, 12136723, 5649507, 3095212, 2197481, 1541822, 1201967, 831487],
                    [48032530, 12096852, 5669281, 3086246, 2205279, 1526406, 1187060, 816447],
                    [47981815, 12065222, 5627945, 3056501, 2186051, 1533407, 1151704, 799838]])

    res = []
    for i in range(len(raw)):
        res.append([])
        for j in range(0,len(raw[i])-1):
            res[i].append(raw[i][j]/float(raw[3][2]))

    # print(res)
    res = np.array(res)
    fig, ax = plt.subplots()
    ax.set_xlabel('SA size', fontsize=26)
    ax.set_ylabel('buffer size', fontsize=26)
    plt.rc('font', size=26)
    plt.tick_params(axis='both', which='major', labelsize=24)
    im, cbar = heatmap(res, y, x, ax=ax,
                   cmap=cmap_name, cbarlabel="Norm. cycle",
                   norm=matplotlib.colors.Normalize(vmin=0, vmax=3))
    # norm=matplotlib.colors.Normalize(vmin=0, vmax=1.0)

    # fig.tight_layout()
    plt.subplots_adjust(left=-0.15, bottom=0.28, right=0.99, top=0.95,
        wspace=0.0, hspace=0.0)
    plt.savefig("profile_sensitivity_cycle.pdf");


def profile_sensitivity_traffic():
    y = ["0.50", "1.00", "1.50", "2.00", "2.50", "3.00"]
    x = ["08x08", "16x16", "24x24", "32x32", "40x40", "48x48", "56x56"]

    raw = np.array([[1018285972.88, 703712889.40, 208859449.84, 210733205.04, 211652352.56, 213421271.28, 202027701.68],  # 0.25
                    [1018285972.88, 587838746.32, 177265966.00, 177332130.48, 180339430.40, 179566518.96, 179356527.60],  # 0.5
                    [1006091671.84, 169861005.56, 169192131.84, 169186049.28, 170086913.28, 169975857.60, 169840661.76],  # 1.0
                    [1056425403.12, 164516141.60, 164356760.96, 164355162.56, 169256808.40, 165145250.24, 165018827.20],  # 1.5
                    [1056425403.12, 162490741.24, 162307659.48, 162624026.20, 168520234.92, 163419704.92, 163297591.00],  # 2.0
                    [189888120.88 , 160464212.64, 160313399.04, 160301715.20, 162175983.36, 161113870.08, 160859664.00],  # 3.0
                    [162370751.6  , 157062329.12, 156246008.64, 156390586.24, 162909212.88, 157219176.32, 157355306.84]]) # 6.0 

    # double buffer-ed
    # GC-Net
    raw = np.array([[1004927671.16, 587838746.32, 177265966.00, 177332130.48, 180339430.40, 179566518.96, 179356527.60, 177865538.48],
                    [1006091671.84, 169861005.56, 169192131.84, 169186049.28, 170086913.28, 169975857.60, 169840661.76, 169799961.60],
                    [756425403.12,  164516141.60, 164356760.96, 164355162.56, 169256808.40, 165145250.24, 165018827.20, 164949724.16],
                    [763769115.00,  162490741.24, 162307659.48, 162624026.20, 168520234.92, 163419704.92, 163297591.00, 163224642.52],
                    [161923934.84,  161092405.88, 160888846.04, 160815548.12, 166723522.60, 161609414.36, 161487857.24, 161414630.36],
                    [189888120.88,  160464212.64, 160313399.04, 160301715.20, 162175983.36, 161113870.08, 160859664.00, 160823792.64]
                    ])
    # [162370751.60 , 157062329.12, 156246008.64, 156390586.24, 162909212.88, 157219176.32, 157355306.84]


    # Flow-Net-C
    raw = np.array([[11715698, 11086322, 11357398, 10997020, 11287467, 10747218, 11088592, 10160431],
                    [9298696, 8695798, 8256739, 8093290, 8277360, 8272555, 8424289, 7895540],
                    [8265297, 8138571, 8069900, 7448125, 7438586, 7383416, 7675217, 7451911],
                    [7383385, 7340166, 7335902, 6644593, 6633072, 6477326, 6877955, 6688953],
                    [7222263, 7178158, 7144430, 6455051, 6483725, 6299954, 6715493, 6515345],
                    [6994974, 6913684, 7095598, 6234930, 6210978, 6018078, 6429285, 6303456]])

    res = []
    for i in range(len(raw)):
        res.append([])
        for j in range(0,len(raw[i])-1):
            res[i].append(raw[i][j]/float(raw[3][2]))

    # print(res)
    res = np.array(res)
    fig, ax = plt.subplots()
    ax.set_xlabel('SA size', fontsize=26)
    ax.set_ylabel('buffer size', fontsize=26)
    plt.rc('font', size=26)
    plt.tick_params(axis='both', which='major', labelsize=24)
    im, cbar = heatmap(res, y, x, ax=ax,
                    cmap=cmap_name, cbarlabel="Norm. energy",
                    norm=matplotlib.colors.Normalize(vmin=0.9, vmax=1.5))
    # norm=matplotlib.colors.Normalize(vmin=0, vmax=2.0)
    # fig.tight_layout()
    plt.subplots_adjust(left=-0.15, bottom=0.28, right=0.99, top=0.95,
        wspace=0.0, hspace=0.0)
    plt.savefig("profile_sensitivity_traffic.pdf");


def profile_sensitivity_speedup():
    y = ["0.50", "1.00", "1.50", "2.00","2.50", "3.00"]
    x = ["08x08", "16x16", "24x24", "32x32", "40x40", "48x48", "56x56"]

    # double buffer
    # GC-Net
    opti =np.array([[1111124424.92, 276631234.00, 169739441.08, 69386352.08, 55471383.76, 46216657.64, 39723227.12, 29354436.04],
                    [1143452566.24, 276289671.96, 169408406.88, 69116178.44, 55297473.72, 46188296.88, 39475960.52, 29020161.04],
                    [1163109930.28, 276208444.76, 169428729.12, 69105705.96, 55260845.72, 46113307.84, 39493259.00, 29030247.16],
                    [1166334800.08, 276143563.68, 169434674.28, 69098600.52, 55241064.76, 46049312.88, 39511974.24, 29029670.84],
                    [1104479570.28, 276120246.48, 169417224.08, 69065728.00, 55232945.20, 46063724.68, 39479170.52, 29033568.76],
                    [1110877199.84, 276149230.32, 169349295.88, 69065460.64, 55234458.80, 46049820.60, 39507994.52, 29023590.16]])

    raw = np.array([[1626070321.64, 407827843.52, 252454117.00, 104454696.64, 85172008.40, 72308262.80, 62906910.32, 55560808.12],
                    [1623093276.88, 406408992.08, 251206044.24, 103383911.76, 84481583.92, 71528297.36, 62993638.96, 54680132.60],
                    [1623193643.08, 405946379.88, 251171798.12, 103264098.48, 84004418.16, 71173429.72, 62701852.76, 54174637.92],
                    [1622778605.84, 405784335.92, 250829634.52, 103301248.56, 84039555.88, 71174358.80, 62418211.44, 53815305.36],
                    [1622275418.04, 405831800.32, 250813542.20, 103280336.56, 83897232.08, 71094926.76, 62427956.56, 53757170.08],
                    [1622347452.56, 405779457.36, 250750680.20, 103140501.20, 83943354.40, 71062222.36, 62280729.64, 53556306.32]])

    # # Flow-Net-C
    opti =np.array([[49493394, 13266263, 6417728, 3524839, 2407647, 1867746, 1301906, 884975],
                    [48689890, 12363012, 5868066, 3329769, 2307832, 1678266, 1239938, 940589],
                    [48195711, 12250135, 5717865, 3200491, 2233979, 1613569, 1178820, 911407],
                    [48112072, 12136723, 5649507, 3095212, 2197481, 1541822, 1201967, 831487],
                    [48032530, 12096852, 5669281, 3086246, 2205279, 1526406, 1187060, 816447],
                    [47981815, 12065222, 5627945, 3056501, 2186051, 1533407, 1151704, 799838]])

    raw = np.array([[73936727, 18669630, 8468355, 4936301, 3491568, 2614627, 2351944, 1884739],
                    [73695799, 18551616, 8249992, 4705880, 3141435, 2360526, 1929731, 1629775],
                    [73702372, 18530089, 8269473, 4700949, 3049597, 2240344, 1789685, 1459346],
                    [73525984, 18463062, 8228841, 4641081, 3014193, 2184528, 1687496, 1423725],
                    [73638045, 18450475, 8240807, 4640820, 3016226, 2173360, 1673123, 1398314],
                    [73462117, 18419370, 8193109, 4637041, 2989710, 2138193, 1674507, 1365300]])

    res = []
    for i in range(len(raw)):
        res.append([])
        for j in range(0,len(raw[i])-1):
            res[i].append(raw[i][j]/float(opti[i][j]))

    # print(res)
    res = np.array(res)
    fig, ax = plt.subplots()
    ax.set_xlabel('PE size', fontsize=26)
    ax.set_ylabel('Buffer size', fontsize=26)
    plt.rc('font', size=26)
    plt.tick_params(axis='both', which='major', labelsize=24)
    im, cbar = heatmap(res, y, x, ax=ax,
                   cmap=cmap_name, cbarlabel="Speedup",
                   norm=matplotlib.colors.Normalize(vmin=1.3, vmax=1.6))
    # norm=matplotlib.colors.Normalize(vmin=0, vmax=1.0)

    # fig.tight_layout()
    plt.subplots_adjust(left=-0.0, bottom=0.30, right=0.95, top=0.95,
        wspace=0.0, hspace=0.0)
    plt.savefig("profile_sensitivity_speedup.pdf");


def profile_sensitivity_saving():
    y = ["0.50", "1.00", "1.50", "2.00", "2.50", "3.00"]
    x = ["08x08", "16x16", "24x24", "32x32", "40x40", "48x48", "56x56"]

    opti = np.array([[1018285972.88, 703712889.40, 208859449.84, 210733205.04, 211652352.56, 213421271.28, 202027701.68],  # 0.25
                    [1018285972.88, 587838746.32, 177265966.00, 177332130.48, 180339430.40, 179566518.96, 179356527.60],  # 0.5
                    [1006091671.84, 169861005.56, 169192131.84, 169186049.28, 170086913.28, 169975857.60, 169840661.76],  # 1.0
                    [256425403.12, 164516141.60, 164356760.96, 164355162.56, 169256808.40, 165145250.24, 165018827.20],  # 1.5
                    [256425403.12, 162490741.24, 162307659.48, 162624026.20, 168520234.92, 163419704.92, 163297591.00],  # 2.0
                    [189888120.88 , 160464212.64, 160313399.04, 160301715.20, 162175983.36, 161113870.08, 160859664.00],  # 3.0
                    [162370751.6  , 157062329.12, 156246008.64, 156390586.24, 162909212.88, 157219176.32, 157355306.84]]) # 6.0 

    # double buffer-ed
    # GC-Net
    raw = np.array([[445039303.48, 445039303.48, 445039303.48, 445117946.68, 445117946.68, 445015065.4, 445322431.36, 444486466.36],
                    [437431680.88, 437431680.88, 437431680.88, 437431680.88, 437431680.88, 437431680.88, 437628718.96, 437431680.88],
                    [433195580.16, 433195580.16, 433195580.16, 433290443.52, 433195580.16, 433195580.16, 433290443.52, 433290443.52],
                    [430330586.24, 430330586.24, 430330586.24, 430330586.24, 430330586.24, 430330586.24, 430330586.24, 430330586.24],
                    [429864840.32, 429864840.32, 429864840.32, 429864840.32, 429864840.32, 429864840.32, 429864840.32, 429864840.32],
                    [428212759.68, 428212759.68, 428212759.68, 428212759.68, 428212759.68, 428212759.68, 428212759.68, 428212759.68]])


    # [162370751.60 , 157062329.12, 156246008.64, 156390586.24, 162909212.88, 157219176.32, 157355306.84]


    # Flow-Net-C
    opti = np.array([[11715698, 11086322, 11357398, 10997020, 11287467, 10747218, 11088592, 10160431],
                    [9298696, 8695798, 8256739, 8093290, 8277360, 8272555, 8424289, 7895540],
                    [8265297, 8138571, 8069900, 7448125, 7438586, 7383416, 7675217, 7451911],
                    [7383385, 7340166, 7335902, 6644593, 6633072, 6477326, 6877955, 6688953],
                    [7222263, 7178158, 7144430, 6455051, 6483725, 6299954, 6715493, 6515345],
                    [6994974, 6913684, 7095598, 6234930, 6210978, 6018078, 6429285, 6303456]])

    raw = np.array([[19858446, 19201511, 21850454, 17567207, 18497570, 17103999, 17015732, 13712083],
                    [12436458, 12697972, 13454865, 12223540, 12433583, 12623094, 12480931, 11322321],
                    [10531202, 11066833, 10787990, 10437388, 10783894, 10715758, 10658099, 9592982],
                    [9213992, 9213992, 10543932, 9305702, 10002337, 9794196, 9548429, 8785438],
                    [9019314, 9180003, 9501382, 9070672, 8909628, 8980441, 8955432, 8587028],
                    [8392331, 8392331, 8392331, 8596481, 8707959, 8511351, 8793266, 8227920]])
    res = []
    for i in range(len(raw)):
        res.append([])
        for j in range(0,len(raw[i])-1):
            res[i].append(1-float(opti[i][j])/raw[i][j])

    # print(res)
    res = np.array(res)
    fig, ax = plt.subplots()
    ax.set_xlabel('PE size', fontsize=26)
    ax.set_ylabel('Buffer size', fontsize=26)
    plt.rc('font', size=26)
    plt.tick_params(axis='both', which='major', labelsize=24)
    im, cbar = heatmap(res, y, x, ax=ax,
                    cmap=cmap_name, cbarlabel="Energy reduction",
                    norm=matplotlib.colors.Normalize(vmin=0.2, vmax=0.5))
    # norm=matplotlib.colors.Normalize(vmin=0, vmax=2.0)
    # fig.tight_layout()
    plt.subplots_adjust(left=-0.0, bottom=0.30, right=0.95, top=0.95,
        wspace=0.0, hspace=0.0)
    plt.savefig("profile_sensitivity_saving.pdf");


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Arguments:
        data       : A 2D numpy array of shape (N,M)
        row_labels : A list or array of length N with the labels
                     for the rows
        col_labels : A list or array of length M with the labels
                     for the columns
    Optional arguments:
        ax         : A matplotlib.axes.Axes instance to which the heatmap
                     is plotted. If not provided, use current axes or
                     create a new one.
        cbar_kw    : A dictionary with arguments to
                     :meth:`matplotlib.Figure.colorbar`.
        cbarlabel  : The label for the colorbar
    All other arguments are directly passed on to the imshow call.
    """

    if not ax:
        ax = plt.gca()

    plt.rc('font', size=20)
    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


# profile_energy()
# profile_speedup()
# profile_sensitivity_cycle()
# profile_sensitivity_traffic()
profile_sensitivity_speedup()
profile_sensitivity_saving()
# profile_speedup_energy()