import os, sys
from pathlib import Path

import cv2
import numpy as np

from load_video import ReadVideo
from load_seg import ReadSegmentation

print(cv2.__version__)
### Load the video ###
video_dir = '/media/ze/Elements/User/Data/SDD/'
v_reader = ReadVideo(video_dir, scenario_idx=3, video_idx=0)
m_in_px = v_reader.real_world_scale(0.036, None)

v_reader.play_video_annotated()

### Load the segmentation ###
root = Path(__file__).parents[1]
label_dir = os.path.join(root, 'Data/SDD_seg/Bookstore/')
s_reader = ReadSegmentation(label_dir)
hulls = s_reader.gen_convex_hulls(label_index=s_reader.class_list.index('Stop'))

### Load the robot ###
x_init = np.array([10,10,0]).reshape(-1,1) * m_in_px
u = np.array([1,0.0]).reshape(-1,1) * m_in_px
ts = 0.1

### Play the video with the robot and segmentation ###
def func_draw(frame):
    cv2.drawContours(frame, hulls, -1, (0,100,255), 2)
extra = {'func_evolve':None, 'func_draw':func_draw, 'ts':ts}
v_reader.play_video_extra(extra=extra)
