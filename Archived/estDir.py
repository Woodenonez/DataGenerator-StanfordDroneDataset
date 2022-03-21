import os, sys, glob
import time

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
def text_with_backgroud(img, text, org, font, scale, color_text=(0,0,0), color_bg=(255,255,255)):
    (txt_w, txt_h) = cv2.getTextSize(text, font, fontScale=scale, thickness=1)[0]
    cv2.rectangle(img, tuple([org[0], org[1]-txt_h-3]), 
                       tuple([org[0]+txt_w, org[1]]), color_bg, cv2.FILLED)
    cv2.putText(img, text, tuple([org[0],org[1]-3]), font, scale, color_text)

### customize
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
# path = '/home/Append/Data/SDD/'
path = '/media/ze/Elements/User/Data/SDD/'
idx_sce = 2 # select scenario
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
df_anno['velocity'] = np.zeros([df_anno.shape[0],1])
df_anno['d1'] = np.zeros([df_anno.shape[0],1])
df_anno['d2'] = np.zeros([df_anno.shape[0],1])

cap = cv2.VideoCapture(files_v[idx_file])
FPS = cap.get(cv2.CAP_PROP_FPS)
print('Showing SDD: "{}({})".'.format(sce_op,idx_file))

### start processing and showing
idx_frame = -1
df_last_frame = df_anno[df_anno['frame']==0]
while(1):
    idx_frame += 1
    df_frame = df_anno[df_anno['frame']==idx_frame]

    ret, frame = cap.read()
    for idx_obj in range(df_frame.shape[0]):
        rec = df_frame.iloc[idx_obj,:] # one record
        [xmin, ymin, xmax, ymax] = rec.loc[['xmin','ymin','xmax','ymax']]
        center = tuple([int((xmin+xmax)/2), int((ymin+ymax)/2)])
        if rec['lost']:
            cv2.circle(frame, center, 2, (0,0,0), 3)
        else:
            if any(list(df_last_frame['ID'].isin([rec['ID']]))):
                last_rec = df_last_frame[df_last_frame['ID']==rec['ID']]
                last_velocity = float(last_rec['velocity'])
                last_center = tuple([int((last_rec['xmin']+last_rec['xmax'])/2), 
                                     int((last_rec['ymin']+last_rec['ymax'])/2)])
                last_direction = np.array([int(last_rec['d1']), int(last_rec['d2'])])

                direction = np.array([center[0]-last_center[0], center[1]-last_center[1]])
                direc_norm = np.linalg.norm(direction)
                velocity = direc_norm*FPS # px/sec
                if velocity == 0:
                    velocity = round(0.2*velocity + 0.8*last_velocity, 2) # smooth velocity
                    direction = last_direction
                else:
                    velocity = round(0.8*velocity + 0.2*last_velocity, 2) # smooth velocity
                    direction = direction/direc_norm*(xmax-xmin)
                    direction = 0.6*direction + 0.4*last_direction
                direction = [int(i) for i in direction]
                df_frame.loc[(df_frame['ID']==rec['ID']) 
                           & (df_frame['frame']==idx_frame), 'velocity'] = velocity
                df_frame.loc[(df_frame['ID']==rec['ID']) 
                           & (df_frame['frame']==idx_frame), 'd1'] = direction[0]
                df_frame.loc[(df_frame['ID']==rec['ID']) 
                           & (df_frame['frame']==idx_frame), 'd2'] = direction[1]
                end_point = tuple([sum(x) for x in zip(center, direction)])
                cv2.arrowedLine(frame, center, end_point, (0,0,255))

            if rec['occluded']:
                cv2.circle(frame, center, int((xmax-xmin)/2), (0,0,255), 2)
            cv2.rectangle(frame, tuple([xmin,ymin]), tuple([xmax,ymax]), (0,255,0), 1)
            text_with_backgroud(frame, rec['label']+':'+str(velocity), tuple([xmin, ymin]), 
                                font, scale=1, color_text=(0,0,0), color_bg=(0,255,0))


    df_last_frame = df_frame

    time.sleep(sleep)
    cv2.imshow('SDD',frame)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break

### clean up
cap.release()
cv2.destroyAllWindows()