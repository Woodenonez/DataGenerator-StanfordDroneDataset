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
path = '/media/ze/Elements/User/Data/SDD/'
idx_sce = 2 # select scenario
idx_file = 0 # select video in the scenario
sleep = 0 # second
BS_approach = 'MOG2' # 'GMG', 'KNN or MOG2'

### obtain operation points
scenario = os.listdir(path+'videos/')
sce_op = scenario[idx_sce] # e.x. 2='bookstore'

path_op = path + 'videos/' + sce_op
path_an = path + 'annotations/' + sce_op
files_v = glob.glob(path_op+'/*/*.*')
files_a = glob.glob(path_an+'/*/*.txt')

if BS_approach == 'KNN':
    BackSub = cv2.createBackgroundSubtractorKNN(history=500,
                                                dist2Threshold=400.0,
                                                detectShadows=1)
elif BS_approach == 'MOG2':
    BackSub = cv2.createBackgroundSubtractorMOG2()
else:
    BackSub = cv2.bgsegm.createBackgroundSubtractorGMG()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

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
    fg_mask = BackSub.apply(frame)

    if BS_approach == 'GMG':
        oframe = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    else:
        oframe = fg_mask

    for idx_obj in range(df_frame.shape[0]):
        rec = df_frame.iloc[idx_obj,:] # one record
        [xmin, ymin, xmax, ymax] = rec.loc[['xmin','ymin','xmax','ymax']]
        center = tuple([int((xmin+xmax)/2), int((ymin+ymax)/2)])
        if not rec['lost']:
            if rec['occluded']:
                cv2.circle(oframe, center, int((xmax-xmin)/2), 255, 2)
            cv2.rectangle(oframe, tuple([xmin,ymin]), tuple([xmax,ymax]), 255, 1)
            text_with_backgroud(oframe, rec['label'], tuple([xmin, ymin]), 
                                font, scale=1, color_text=255, color_bg=0)


    df_last_frame = df_frame

    time.sleep(sleep)
    cv2.imshow('SDD',oframe)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break

### clean up
cap.release()
cv2.destroyAllWindows()