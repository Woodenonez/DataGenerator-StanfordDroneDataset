import os, sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from util import utils_data
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
save_path = os.path.join(root_dir, 'Data/SDD_Crossing_Ped5s')

past = 4
img_saving_period = 10  # 30 FPS -> 3 FPS
dataset_gen_period = 1

minT = 15
maxT = 15 # 3 FPS -> 5 s

test_split = 0.1 # if we split trajectories or not

# all_video_idx = [list(range(7)), list(range(4)), list(range(5)), list(range(9)), 
#                  list(range(15)), list(range(4)), list(range(12)), list(range(4))]

all_video_idx = [[],[],[],[],[0,1,2,3,7,8,9,10,11,12]]

for scenario_idx in range(len(all_video_idx)):
    for video_idx in all_video_idx[scenario_idx]:
        verbose = False
        if (scenario_idx==0) and (video_idx==0):
            verbose = True
        video_reader = ReadVideo(data_dir, scenario_idx=scenario_idx, video_idx=video_idx, verbose=verbose)
        if test_split == 0:
            utils_data.save_SDD_data(video_reader, save_path=save_path, period=img_saving_period)
        else:
            utils_data.save_SDD_data_part(video_reader, save_path=save_path, test_split=test_split, period=img_saving_period)
        print(f'Scenario {scenario_idx}-{video_idx} images generated!')

if test_split == 0:
    utils_data.gather_all_data(save_path, past, minT=minT, maxT=maxT, period=dataset_gen_period) # go through all the obj folders and put them together in one CSV    
else:
    utils_data.gather_all_data(save_path+'_training', past, minT=minT, maxT=maxT, period=dataset_gen_period) # go through all the obj folders and put them together in one CSV
    utils_data.gather_all_data(save_path+'_testing', past, minT=minT, maxT=maxT, period=dataset_gen_period) # go through all the obj folders and put them together in one CSV
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