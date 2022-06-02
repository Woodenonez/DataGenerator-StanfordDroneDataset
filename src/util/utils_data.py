from errno import ELIBBAD
import os, sys
import glob
from pathlib import Path

import numpy as np
import pandas as pd

import cv2

'''

'''

def gather_all_data_position(data_dir:str, past:int, minT:int, maxT:int, period=1, save_dir=None) -> None:
    # data_dir  -  video idx - imgs&csv
    if save_dir is None:
        save_dir = data_dir

    column_name = [f'p{i}' for i in range(0, past+1)] + ['t', 'id', 'index', 'T']
    df_all = pd.DataFrame(columns=column_name)
    video_folders = [x for x in os.listdir(data_dir) if '.' not in x]
    vcnt = 0 # cnt for videos
    for vf in video_folders:
        vcnt += 1
        csv_name = glob.glob(os.path.join(data_dir, vf, '*.csv'))
        df_video = pd.read_csv(csv_name[0])
        print(f'\rProcess: video-{vcnt}/{len(video_folders)}', end='    ')
        all_obj = df_video['id'].unique()
        
        for i in range(len(all_obj)):
            obj_id = all_obj[i]
            df_obj = df_video[df_video['id'] == obj_id]
            for T in range(minT,maxT+1):
                sample_list = []
                for i in range(len(df_obj)-past*period-T): # each sample
                    sample = []
                    ################## Sample START ##################
                    for j in range(past+1):
                        obj_past = f'{df_obj.iloc[i+j*period]["x"]}_{df_obj.iloc[i+j*period]["y"]}_{df_obj.iloc[i+j*period]["t"]}'
                        sample.append(obj_past)
                    sample.append(df_obj.iloc[i+past]['t'])
                    sample.append(obj_id)
                    sample.append(df_obj.iloc[i+past+T]['index'])
                    sample.append(f'{df_obj.iloc[i+past+T]["x"]}_{df_obj.iloc[i+past+T]["y"]}')
                    ################## Sample E N D ##################
                    sample_list.append(sample)
                df_T = pd.DataFrame(sample_list, columns=df_all.columns)
                df_all = pd.concat([df_all, df_T], ignore_index=True)
    df_all.to_csv(os.path.join(save_dir, 'all_data.csv'), index=False)

def gather_all_data_trajectory(data_dir:str, past:int, minT:int, maxT:int, period=1, save_dir=None) -> None:
    # data_dir  -  video idx - imgs&csv
    if save_dir is None:
        save_dir = data_dir

    column_name = [f'p{i}' for i in range(0,(past+1))] + ['t', 'id', 'index'] + [f'T{i}' for i in range(minT, maxT+1)]
    df_all = pd.DataFrame(columns=column_name)
    video_folders = [x for x in os.listdir(data_dir) if '.' not in x]
    vcnt = 0 # cnt for videos
    for vf in video_folders:
        vcnt += 1
        csv_name = glob.glob(os.path.join(data_dir, vf, '*.csv'))
        df_video = pd.read_csv(csv_name[0])
        print(f'\rProcess: video-{vcnt}/{len(video_folders)}', end='    ')
        all_obj = df_video['id'].unique()
        
        for i in range(len(all_obj)):
            obj_id = all_obj[i]
            df_obj = df_video[df_video['id'] == obj_id]

            sample_list = []
            for i in range(len(df_obj)-past*period-maxT): # each sample
                sample = []
                ################## Sample START ##################
                for j in range(past+1):
                    obj_past = f'{df_obj.iloc[i+j*period]["x"]}_{df_obj.iloc[i+j*period]["y"]}_{df_obj.iloc[i+j*period]["t"]}'
                    sample.append(obj_past)
                sample.append(df_obj.iloc[i+past]['t'])
                sample.append(obj_id)
                sample.append(df_obj.iloc[i+past+maxT]['index'])

                for T in range(minT, maxT+1):
                    sample.append(f'{df_obj.iloc[i+past+T]["x"]}_{df_obj.iloc[i+past+T]["y"]}')
                ################## Sample E N D ##################
                sample_list.append(sample)
            df_T = pd.DataFrame(sample_list, columns=df_all.columns)
            df_all = pd.concat([df_all, df_T], ignore_index=True)


    df_all.to_csv(os.path.join(save_dir, 'all_data.csv'), index=False)


def save_SDD_data(video_reader, save_path:str, period:int=1, resize_label:bool=True, tr_name=None, tr=None) -> None:  # dynamic env
    # scenario indices: 0~7 (bookstore, coupa, deathCircle, gates, hyang,  little, nexus,  quad)
    # video indices:         0:0~6,     1:0~3, 2:0~4,       3:0~8, 4:0~14, 5:0~3,  6:0~11, 7:0~3
    #                        1          1      1            2      2       1       2       0
    scene_name = video_reader.scenario_name
    video_name = video_reader.video_name

    t_list = []   # time or time step
    id_list = []
    idx_list = [] # more information (e.g. scene and video index)
    x_list = []   # x coordinate
    y_list = []   # y coordinate

    cnt = 0
    period_cnt = 0
    swapHW = False
    while(1):
        try:
            frame, df_frame = video_reader.read_frame_clean()
            cnt += 1
            print(f'\rAnalyzing: Frame {cnt}/{video_reader.info("nframes")}', end='   ')
            if period_cnt < period-1:
                period_cnt += 1
                continue
            else:
                period_cnt = 0
            ### Ensure H<W
            if frame.shape[0] > frame.shape[1]:
                frame = frame.swapaxes(0,1)
                swapHW = True
            ### Resize
            original_size = frame.shape[:2][::-1] # WxH
            new_size = (595, 326)                 # WxH, TrajNet:595x326 [or (576, 320)]
            frame = cv2.resize(frame, new_size)   # HxWxC
        except:
            break

        ### Screening conditions
        if df_frame.shape[0] < 1:
            continue
        df_frame = df_frame.loc[df_frame['label']=='Pedestrian'] # only pedestrians!!!

        ### Start to save trajectories
        df_frame = df_frame.loc[df_frame['lost'] != 1]
        df_x = (df_frame['xmin'] + df_frame['xmax']) // 2
        df_y = (df_frame['ymin'] + df_frame['ymax']) // 2
        if swapHW: # if H W are swapped
            df_x, df_y = df_y, df_x
        if resize_label:
            df_x *= new_size[0]/original_size[0]
            df_y *= new_size[1]/original_size[1]
        
        t_list += [cnt] * len(df_frame)
        id_list += df_frame['ID'].values.tolist()
        x_list_temp = df_x.values.tolist()
        y_list_temp = df_y.values.tolist()
        idx_list += [f'{scene_name}_{video_name}_{tr_name}'] * len(df_frame)

        if tr is not None:
            if '90' in tr_name:
                k = 1
            elif '180' in tr_name:
                k = 2
            elif '270' in tr_name:
                k = 3
            else:
                k = 0
            frame, xy_np = tr(frame, np.array([x_list_temp, y_list_temp]).transpose(), k=k)
            x_list_temp = xy_np[:,0].tolist()
            y_list_temp = xy_np[:,1].tolist()
        x_list += x_list_temp
        y_list += y_list_temp

        ### save images and dataframe
        if save_path is None:
            cv2.imshow("frame", frame)
            cv2.waitKey()
        else:
            folder = os.path.join(save_path,f'{scene_name}_{video_name}_{tr_name}/')
            Path(folder).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(os.path.join(folder,f'{cnt}.jpg'), frame)

    df = pd.DataFrame({'t':t_list, 'id':id_list, 'index':idx_list, 'x':x_list, 'y':y_list}).sort_values(by='t', ignore_index=True)
    df.to_csv(os.path.join(save_path, f'{scene_name}_{video_name}_{tr_name}', f'{original_size[0]}_{original_size[1]}.csv'), index=False)
            
    video_reader.cap.release()
    print()

def save_SDD_data_split(video_reader, save_path:str, test_split:float, period:int=1, resize_label:bool=True, tr_name=None, tr=None) -> None:  # dynamic env
    # scenario indices: 0~7 (bookstore, coupa, deathCircle, gates, hyang,  little, nexus,  quad)
    # video indices:         0:0~6,     1:0~3, 2:0~4,       3:0~8, 4:0~14, 5:0~3,  6:0~11, 7:0~3
    #                        1          1      1            2      2       1       2       0
    assert(0<test_split<1),('Split ratio must be in (0,1).')
    scene_name = video_reader.scenario_name
    video_name = video_reader.video_name

    ### Split into training and testing sets
    all_id = np.unique(video_reader.df_data['ID'].values)
    np.random.seed(0)
    np.random.shuffle(all_id)
    id_for_training, id_for_testing = np.split(all_id, [int((len(all_id)-1)*(1-test_split))])

    t_list_train = []   # time or time step
    id_list_train = []
    idx_list_train = [] # more information (e.g. scene and video index)
    x_list_train = []   # x coordinate
    y_list_train = []   # y coordinate
    
    t_list_test = []   # time or time step
    id_list_test = []
    idx_list_test = [] # more information (e.g. scene and video index)
    x_list_test = []   # x coordinate
    y_list_test = []   # y coordinate
    
    cnt = 0
    period_cnt = 0
    swapHW = False
    while(1):
        try:
            frame, df_frame = video_reader.read_frame_clean()
            cnt += 1
            print(f'\rAnalyzing: Frame {cnt}/{video_reader.info("nframes")}', end='   ')
            if period_cnt < period-1:
                period_cnt += 1
                continue
            else:
                period_cnt = 0
            ### Ensure H<W
            if frame.shape[0] > frame.shape[1]:
                frame = frame.swapaxes(0,1)
                swapHW = True
            ### Resize
            original_size = frame.shape[:2][::-1] # WxH
            new_size = (595, 326)                 # WxH, TrajNet:595x326
            frame = cv2.resize(frame, new_size)
        except:
            break

        ### Screening conditions
        if df_frame.shape[0] < 1:
            continue
        df_frame = df_frame.loc[df_frame['label']=='Pedestrian'] # only pedestrians!!!

        no_testing = True
        no_training = True
        ### Start to save trajectories
        for j in range(len(df_frame)):
            df_obj = df_frame.iloc[j,:]
            if df_obj['lost'] == 1: # if the object is lost, don't save it
                continue

            x = (df_obj['xmin'] + df_obj['xmax']) // 2
            y = (df_obj['ymin'] + df_obj['ymax']) // 2
            if swapHW: # if H W are swapped
                x,y = y,x
            if resize_label:
                x *= new_size[0]/original_size[0]
                y *= new_size[1]/original_size[1]

            if df_obj["ID"] in id_for_training:
                no_training = False
                id_list_train.append(df_obj["ID"])
                t_list_train.append(cnt)
                x_list_train.append(x)
                y_list_train.append(y)
                idx_list_train.append(f'{scene_name}_{video_name}_{tr_name}')
            elif df_obj["ID"] in id_for_testing:
                no_testing = False
                id_list_test.append(df_obj["ID"])
                t_list_test.append(cnt)
                x_list_test.append(x)
                y_list_test.append(y)
                idx_list_test.append(f'{scene_name}_{video_name}_{tr_name}')
            else:
                continue

        if tr is not None:
            if '90' in tr_name:
                k = 1
            elif '180' in tr_name:
                k = 2
            elif '270' in tr_name:
                k = 3
            else:
                k = 0
            frame, xy_np = tr(frame, np.array([x_list_train, y_list_train]).transpose(), k=k)
            x_list_train = xy_np[:,0].tolist()
            y_list_train = xy_np[:,1].tolist()
            _, xy_np = tr(np.zeros_like(frame), np.array([x_list_test, y_list_test]).transpose(), k=k)
            x_list_test = xy_np[:,0].tolist()
            y_list_test = xy_np[:,1].tolist()

        if save_path is None:
            cv2.imshow("frame", frame)
            cv2.waitKey()
        else:
            if not no_training:
                folder = os.path.join(save_path+'_training',f'{scene_name}_{video_name}_{tr_name}/')
                Path(folder).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(os.path.join(folder,f'{cnt}.jpg'), frame)
            if not no_testing:
                folder = os.path.join(save_path+'_testing',f'{scene_name}_{video_name}_{tr_name}/')
                Path(folder).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(os.path.join(folder,f'{cnt}.jpg'), frame)

        if t_list_train:
            df_training = pd.DataFrame({'t':t_list_train, 'id':id_list_train, 'index':idx_list_train, 'x':x_list_train, 'y':y_list_train}).sort_values(by='t', ignore_index=True)
            df_training.to_csv(os.path.join(save_path+'_training', f'{scene_name}_{video_name}_{tr_name}', f'{original_size[0]}_{original_size[1]}.csv'), index=False)

        if t_list_test:
            df_testing = pd.DataFrame({'t':t_list_test, 'id':id_list_test, 'index':idx_list_test, 'x':x_list_test, 'y':y_list_test}).sort_values(by='t', ignore_index=True)
            df_testing.to_csv(os.path.join(save_path+'_testing', f'{scene_name}_{video_name}_{tr_name}', f'{original_size[0]}_{original_size[1]}.csv'), index=False)

    video_reader.cap.release()
    print()

def save_SDD_data_seg(seg_reader, save_path:str, period:int=1, resize_label:bool=True, tr_name=None, tr=None) -> None:  # static env
    # scenario indices: 0~7 (bookstore, coupa, deathCircle, gates, hyang,  little, nexus,  quad)
    # video indices:         0:0~6,     1:0~3, 2:0~4,       3:0~8, 4:0~14, 5:0~3,  6:0~11, 7:0~3
    #                        1          1      1            2      2       1       2       0
    scene_name = seg_reader.scenario_name
    video_name = seg_reader.video_name
    img_seg, img_ref = (seg_reader.img_seg, seg_reader.img_ref)

    t_list = []   # time or time step
    id_list = []
    idx_list = [] # more information (e.g. scene and video index)
    x_list = []   # x coordinate
    y_list = []   # y coordinate

    swapHW = False
    ### Ensure H<W
    if img_ref.shape[0] > img_ref.shape[1]:
        img_ref = img_ref.swapaxes(0,1)
        img_seg = img_seg.swapaxes(0,1)
        swapHW = True
    ### Resize
    original_size = img_ref.shape[:2][::-1] # WxH
    new_size = (595, 326)                   # WxH, TrajNet:595x326 [or (576, 320)]
    img_ref = cv2.resize(img_ref, new_size)
    img_seg = cv2.resize(img_seg, new_size)

    anno = seg_reader.df_data # ['ID','xmin','ymin','xmax','ymax','frame','lost','occluded','generated','label']
    for tk in range(0, seg_reader.nframes, period):
        df_frame = anno.loc[anno['frame']==tk]

        ### Screening conditions
        if df_frame.shape[0] < 1:
            continue
        df_frame = df_frame.loc[df_frame['label']=='Pedestrian'] # only pedestrians!!!

        ### Start to save trajectories
        df_frame = df_frame.loc[df_frame['lost'] != 1]
        df_x = (df_frame['xmin'] + df_frame['xmax']) // 2
        df_y = (df_frame['ymin'] + df_frame['ymax']) // 2
        if swapHW: # if H W are swapped
            df_x, df_y = df_y, df_x
        if resize_label:
            df_x *= new_size[0]/original_size[0]
            df_y *= new_size[1]/original_size[1]

        t_list += [tk] * len(df_frame)
        id_list += df_frame['ID'].values.tolist()
        idx_list += [f'{scene_name}_{video_name}_{tr_name}'] * len(df_frame)
        x_list_temp = df_x.values.tolist()
        y_list_temp = df_y.values.tolist()

        if tr is not None:
            if '90' in tr_name:
                k = 1
            elif '180' in tr_name:
                k = 2
            elif '270' in tr_name:
                k = 3
            else:
                k = 0
            _, xy_np = tr(np.zeros_like(img_ref), np.array([x_list_temp, y_list_temp]).transpose(), k=k)
            x_list_temp = xy_np[:,0].tolist()
            y_list_temp = xy_np[:,1].tolist()
        x_list += x_list_temp
        y_list += y_list_temp

    folder = os.path.join(save_path, f'{scene_name}_{video_name}_{tr_name}/')
    Path(folder).mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({'t':t_list, 'id':id_list, 'index':idx_list, 'x':x_list, 'y':y_list}).sort_values(by='t', ignore_index=True)
    df.to_csv(os.path.join(folder, f'{original_size[0]}_{original_size[1]}.csv'), index=False)

    if tr is not None:
        if '90' in tr_name:
            k = 1
        elif '180' in tr_name:
            k = 2
        elif '270' in tr_name:
            k = 3
        else:
            k = 0
        img_ref, _ = tr(img_ref, np.array([x_list, y_list]).transpose(), k=k)
        img_seg, _ = tr(img_seg, np.array([x_list, y_list]).transpose(), k=k)

    cv2.imwrite(os.path.join(folder,'reference.jpg'), img_ref)
    cv2.imwrite(os.path.join(folder,'label.png'), img_seg)

    cv2.imwrite(os.path.join(save_path,'reference.jpg'), img_ref) # for later checking shape


