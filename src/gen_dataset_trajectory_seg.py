import os, sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from util import utils_data
import load_seg

print('\nGenerating dataset...')

'''
# scenario indices: 0~7 (bookstore, coupa, deathCircle, gates, hyang,   little, nexus,  quad)
# video indices:         0:0~6,     1:0~3, 2:0~4,       3:0~8, 4:0~14,  5:0~3,  6:0~11, 7:0~3
#                        1:2        1:2    1:0          2:3,6  2:13,14  1:2     2:3,11  0
# Need H-W swap: scene 2, 3, 5, 6
# test_video_idx = [[2], [2], [0], [3,6], [13,14], [2], [11], []]
'''

root_dir = Path(__file__).resolve().parents[1]
data_dir = os.path.join(root_dir, 'Data/segmentation_source')
save_path = os.path.join(root_dir, 'Data/ped/SDD_seg_train')

past = 8 # 2.5 FPS -> 3.2 s
img_saving_period = 12  # 30 FPS -> 2.5 FPS
dataset_gen_period = 1

minT = 1
maxT = 12 # 2.5 FPS -> 4.8 s

test_split = 0 # if we split trajectories or not

scenario_name_list = ['bookstore', 'coupa', 'deathCircle', 'gates', 'hyang', 'little', 'nexus', 'quad']
video_name_dict_train = {'bookstore':     [f'video{i}' for i in [0,1,3,4,5,6]], 
                        'coupa':         [f'video{i}' for i in [0,1,3]], 
                        'deathCircle':   [f'video{i}' for i in [1,2,3,4]], 
                        'gates':         [f'video{i}' for i in [0,1,2,4,5,7,8]], 
                        'hyang':         [f'video{i}' for i in [0,1,2,3,4,5,6,7,8,9,10,11,12]], 
                        'little':        [f'video{i}' for i in [0,1,3]], 
                        'nexus':         [f'video{i}' for i in [0,1,2,4,5,6,7,8,9,10]], 
                        'quad':          [f'video{i}' for i in [0,1,2,3]]}
video_name_dict_test  = {'bookstore':     [f'video{i}' for i in [2]], 
                        'coupa':         [f'video{i}' for i in [2]], 
                        'deathCircle':   [f'video{i}' for i in [0]], 
                        'gates':         [f'video{i}' for i in [3, 6]], 
                        'hyang':         [f'video{i}' for i in [13, 14]], 
                        'little':        [f'video{i}' for i in [2]], 
                        'nexus':         [f'video{i}' for i in [3, 11]], 
                        'quad':          []}
video_name_dict = video_name_dict_train

nvideos = sum([len(value) for _, value in video_name_dict.items()])

verbose = True
cnt = 0
for scenario_name in scenario_name_list:
    for video_name in video_name_dict[scenario_name]:
        cnt += 1
        print(f'{cnt}/{nvideos}')
        seg_reader = load_seg.ReadSegResult(data_dir, scenario_name=scenario_name, video_name=video_name, verbose=verbose)
        if test_split == 0:
            utils_data.save_SDD_data_seg(seg_reader, save_path=save_path, period=img_saving_period)
        else:
            raise ModuleNotFoundError
        print(f'Scenario {scenario_name}/{video_name} data generated!\n')

if test_split == 0:
    utils_data.gather_all_data_traj(save_path, past, minT=minT, maxT=maxT, period=dataset_gen_period) # go through all the obj folders and put them together in one CSV    
else:
    utils_data.gather_all_data_traj(save_path+'_train', past, minT=minT, maxT=maxT, period=dataset_gen_period) # go through all the obj folders and put them together in one CSV
    utils_data.gather_all_data_traj(save_path+'_test', past, minT=minT, maxT=maxT, period=dataset_gen_period) # go through all the obj folders and put them together in one CSV
print('Final CSV generated!')

sys.exit(0)
