import os, sys, glob
import time

from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd

'''
    1   Track ID.
    2~5 xmin.ymin.xmax.ymax.
    6   frame (start from 0).
    7   lost. If 1, the annotation is outside of the view screen.
    8   occluded. If 1, the annotation is occluded.
    9   generated. If 1, the annotation was automatically interpolated.
    10  label (enclosed in quotation marks).
'''

### subfunction
def text_with_backgroud(img, text, org, font, scale, 
                        color_text=(0,0,0), color_bg=(255,255,255)):
    (txt_w, txt_h) = cv2.getTextSize(text, font, fontScale=scale, thickness=1)[0]
    cv2.rectangle(img, tuple([org[0], org[1]-txt_h-3]), 
                       tuple([org[0]+txt_w, org[1]]), color_bg, cv2.FILLED)
    cv2.putText(img, text, tuple([org[0],org[1]-3]), font, scale, color_text)

### customize
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
path = '/media/ze/Elements/User/Data/SDD/'
idx_sce = 0 # select scenario
idx_file = 0 # select video in the scenario
sleep = 0 # second

### obtain operation points
scenario = os.listdir(path+'videos/')
sce_op = scenario[idx_sce] # e.x. 2='bookstore'

path_op = path + 'videos/' + sce_op
path_an = path + 'annotations/' + sce_op
files_v = glob.glob(path_op+'/*/*.*')
files_a = glob.glob(path_an+'/*/*.txt')

### get the annotation and video
df_anno = pd.read_csv(files_a[idx_file], sep=' ', header=None,
                   names=['ID','xmin','ymin','xmax','ymax',
                   'frame','lost','occluded','generated','label'])
category = df_anno['label'].unique() # ['Skater' 'Pedestrian' 'Biker' 'Cart' 'Car' 'Bus']
df_anno['velocity'] = np.zeros([df_anno.shape[0],1])
df_anno['d1'] = np.zeros([df_anno.shape[0],1])
df_anno['d2'] = np.zeros([df_anno.shape[0],1])

cap = cv2.VideoCapture(files_v[idx_file])
FPS = cap.get(cv2.CAP_PROP_FPS)
print('Showing SDD: "{}({})".'.format(sce_op,idx_file))

### start processing and showing
idx_frame = -1
df_last_frame = df_anno[df_anno['frame']==0]
v_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
v_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_memory = np.zeros([v_height, v_width, len(category)])
last_frame = []
while(1):
    idx_frame += 1
    df_frame = df_anno[df_anno['frame']==idx_frame]

    ret, frame = cap.read()
    if not ret: 
        frame = last_frame
        break
    
    for idx_obj in range(df_frame.shape[0]):
        rec = df_frame.iloc[idx_obj,:] # one record
        if not rec['lost']:
            [xmin, ymin, xmax, ymax] = rec.loc[['xmin','ymin','xmax','ymax']]
            center = tuple([int((xmin+xmax)/2), int((ymin+ymax)/2)])
            if rec['label'] == 'Biker' or 'Pedestrian':
                if rec['occluded']:
                    cv2.circle(frame, center, int((xmax-xmin)/2), (0,0,255), 2)
                cv2.rectangle(frame, tuple([xmin,ymin]), tuple([xmax,ymax]), (0,255,0), 1)
                text_with_backgroud(frame, rec['label'], tuple([xmin, ymin]), 
                                    font, scale=1, color_text=(0,0,0), color_bg=(0,255,0))
            idx_cate = np.where(category==rec['label'])[0][0]
            frame_memory[center[1],center[0],idx_cate] = idx_cate+1            

    last_frame = frame
    time.sleep(sleep)
    cv2.namedWindow('SDD')
    cv2.imshow('SDD',frame)
    cv2.namedWindow('Mem')
    cv2.imshow('Mem',frame_memory[:,:,2])
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break

### clean up
cap.release()
cv2.destroyAllWindows()

### draw memory
# scat = np.where(frame_memory[:,:,2]!=0)
# plt.scatter(scat[0],scat[1])
# plt.axis([0,v_height,0,v_width])
# plt.gca().invert_yaxis()
# plt.show()
img1 = np.reshape(frame_memory[:,:,2]*200, (v_height,v_width))
img2 = np.reshape(frame_memory[:,:,1]*100, (v_height,v_width))
img3 = np.array(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))

dst = 0.2*img1 + 0.2*img2 + 0.6*img3
plt.imshow(dst, cmap='gray')
plt.show()