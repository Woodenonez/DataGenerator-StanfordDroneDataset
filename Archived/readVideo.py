import os, sys, glob

import numpy as np
import pandas as pd
import cv2

'''
    1   Track ID. All rows with the same ID belong to the same path.
    2   xmin. The top left x-coordinate of the bounding box.
    3   ymin. The top left y-coordinate of the bounding box.
    4   xmax. The bottom right x-coordinate of the bounding box.
    5   ymax. The bottom right y-coordinate of the bounding box.
    6   frame. The frame that this annotation represents.
    7   lost. If 1, the annotation is outside of the view screen.
    8   occluded. If 1, the annotation is occluded.
    9   generated. If 1, the annotation was automatically interpolated.
    10  label. The label for this annotation, enclosed in quotation marks.
'''

def text_with_backgroud(img, text, org, font, scale, color_text=(0,0,0), color_bg=(255,255,255)):
    (txt_w, txt_h) = cv2.getTextSize(text, font, fontScale=scale, thickness=1)[0]
    cv2.rectangle(img, tuple([org[0], org[1]-txt_h-3]), tuple([org[0]+txt_w, org[1]]), color_bg, cv2.FILLED)
    cv2.putText(img, text, tuple([org[0],org[1]-3]), font, scale, color_text)


font = cv2.FONT_HERSHEY_COMPLEX_SMALL
# path = '/home/Append/Data/SDD/' # change the path to yours
path = '/media/ze/Elements/User/Data/SDD/'
show_label = 1

scenario = os.listdir(path+'videos/')

sce_op = scenario[0]

path_op = path + 'videos/' + sce_op
path_an = path + 'annotations/' + sce_op
files_v = glob.glob(path_op+'/*/*.*')
files_a = glob.glob(path_an+'/*/*.txt')

idx_file = 0

cap = cv2.VideoCapture(files_v[idx_file])

anno = files_a[idx_file]
df_data = pd.read_csv(anno, sep=' ', header=None,
                   names=['ID','xmin','ymin','xmax','ymax',
                   'frame','lost','occluded','generated','label'])

print('Showing SDD: "{}".'.format(sce_op))

idx_frame = -1
while(1):
    idx_frame += 1
    df_frame = df_data[df_data['frame']==idx_frame]

    ret, frame = cap.read()
    if not ret: break

    if show_label:
        for frm in range(df_frame.shape[0]):
            rec = df_frame.iloc[frm,:]
            [xmin, ymin, xmax, ymax] = rec.loc[['xmin','ymin','xmax','ymax']]
            center = tuple([int((xmin+xmax)/2), int((ymin+ymax)/2)])
            if rec['lost']:
                cv2.circle(frame, center, 2, (0,0,0), 3)
            else:
                if rec['occluded']:
                    cv2.circle(frame, center, int((xmax-xmin)/2), (0,0,255), 2)
                cv2.rectangle(frame, tuple([xmin,ymin]), tuple([xmax,ymax]), (0,255,0), 1)
                text_with_backgroud(frame, rec['label'], tuple([xmin, ymin]), 
                                    font, scale=1, color_text=(0,0,0), color_bg=(0,255,0))
    cv2.circle(frame, tuple([5,15]), 5, (0,0,255), 2)
    cv2.rectangle(frame, tuple([1,30]), tuple([10,40]), (0,255,0), 1)                     
    cv2.putText(frame, ': occluded', tuple([12,20]), font, 1, (0,0,255))
    cv2.putText(frame, ': detected', tuple([12,40]), font, 1, (0,255,0))
    # frame = cv2.resize(frame,(int(frame.shape[1]/2),int(frame.shape[0]/2)),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
    cv2.imshow('SDD',frame)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()