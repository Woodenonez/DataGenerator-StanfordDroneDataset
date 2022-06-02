import os, sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from util import utils_data
from util import utils_preprocess
from load_video import ReadVideo

print('\nGenerating dataset...')

'''
# scenario indices: 0~7 (bookstore, coupa, deathCircle, gates, hyang,   little, nexus,  quad)
# video indices:         0:0~6,     1:0~3, 2:0~4,       3:0~8, 4:0~14,  5:0~3,  6:0~11, 7:0~3
#                        1:2        1:2    1:0          2:3,6  2:13,14  1:2     2:3,11  0
# Need H-W swap: scene 2, 3, 5, 6
# test_video_idx = [[2], [2], [0], [3,6], [13,14], [2], [11], []]
'''

root_dir = Path(__file__).resolve().parents[1]
data_dir = '/media/ze/Elements/User/Data/SDD/'
save_path = os.path.join(root_dir, 'Data/SDD_15_train')
# tr_name_list = ['',   'fliplr',                'rot180',             'rot180_fliplr']
# tr_list      = [None, utils_preprocess.fliplr, utils_preprocess.rot, utils_preprocess.rot_n_fliplr]
tr_name_list = ['']
tr_list = [None]

past = 4
img_saving_period = 10  # original 30 FPS -> 3 FPS
dataset_gen_period = 1

minT = 15
maxT = 15 # 3 FPS -> 5 s

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
video_name_dict = video_name_dict_test

# scenario_name_list = ['hyang']
# video_name_dict = {'hyang': [f'video{i}' for i in [0]]}

nvideos = sum([len(value) for _, value in video_name_dict.items()]) * len(tr_list)

# verbose = True
# cnt = 0
# for scenario_name in scenario_name_list:
#     for video_name in video_name_dict[scenario_name]:
#         for tr_name, tr in zip(tr_name_list, tr_list):
#             cnt += 1
#             print(f'{cnt}/{nvideos}')
#             video_reader = ReadVideo(data_dir, scenario_name=scenario_name, video_name=video_name, verbose=verbose)
#             verbose = False
#             if test_split == 0:
#                 utils_data.save_SDD_data(video_reader, save_path=save_path, period=img_saving_period, tr_name=tr_name, tr=tr)
#             else:
#                 utils_data.save_SDD_data_split(video_reader, save_path=save_path, test_split=test_split, period=img_saving_period, tr_name=tr_name, tr=tr)
#             print(f'Scenario {scenario_name}-{video_name} images generated!')

if test_split == 0:
    utils_data.gather_all_data_position(save_path, past, minT=minT, maxT=maxT, period=dataset_gen_period) # go through all the obj folders and put them together in one CSV    
else:
    utils_data.gather_all_data_position(save_path+'_training', past, minT=minT, maxT=maxT, period=dataset_gen_period) # go through all the obj folders and put them together in one CSV
    utils_data.gather_all_data_position(save_path+'_testing',  past, minT=minT, maxT=maxT, period=dataset_gen_period) # go through all the obj folders and put them together in one CSV
print('Final CSV generated!')

sys.exit(0)

### Trajectory example
anno = os.path.join(data_dir, 'annotations/bookstore/video0/annotations.txt')
df_terms = ['ID','xmin','ymin','xmax','ymax','frame','lost','occluded','generated','label']
#            0    213    1038   241    1072   10000   1      0          0          "Biker"
df_data = pd.read_csv(anno, sep=' ', header=None, names=df_terms)
df_data = df_data[['ID','xmin','ymin','xmax','ymax','frame','label']]
print(list(df_data['label'].unique()))

df = df_data.loc[df_data['ID']==15].sort_values(by=['frame'], ascending = True)
x = (df['xmin'].to_numpy() + df['xmax'].to_numpy()) / 2
y = (df['ymin'].to_numpy() + df['ymax'].to_numpy()) / 2
traj = np.hstack([x.reshape(-1,1), y.reshape(-1,1)])
label = df['label'].iloc[0]
print(traj)

plt.imshow(plt.imread('/home/ze/Documents/Code/Python_Code/MotionPredRob_SDD/src/reference.jpg'))
plt.plot(traj[:,0],traj[:,1],'.')
plt.show()