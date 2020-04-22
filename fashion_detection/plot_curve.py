import os
import matplotlib.pyplot as plt
import numpy as np

with open('./logs/map.log', 'r') as f :
    data=f.readlines()
    data=[line.strip() for line in data]
    data=[line.split('  ') for line in data]
    map_res=[float(line[1]) for line in data]
def smooth_curve(points, factor=0.65):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points
if __name__ == '__main__':
    _,ax=plt.subplots(1,1,figsize=(10,8))
    # Summarize history for accuracy
    ax.plot(range(1, len(map_res) + 1), smooth_curve(map_res,factor=0.3),'--',label='map')
    ax.set_title('MAP Curve')
    ax.set_ylabel('MAP')
    ax.set_xlabel('Epoch')
    ax.set_xticks(np.arange(1, len(map_res) + 1))
    # ax.set_yticks(np.arange(0.6,1.0,0.05))
    ax.legend(loc='best')
    plt.savefig('map.jpg')