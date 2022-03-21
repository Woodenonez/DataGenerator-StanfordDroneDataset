import os, sys
import glob
from pathlib import Path

import numpy as np
import pandas as pd

import cv2

'''

'''

def gather_all_data(data_dir: str, past: int, minT: int, maxT: int, period=1, save_dir=None) -> None:
    # data_dir  -  video idx - imgs&csv
    if save_dir is None:
        save_dir = data_dir

    column_name = [f't{i}' for i in range(0, past+1)] + ['ID', 'x', 'y', 'T', 'index']
    df_all = pd.DataFrame(columns=column_name)

    video_folders = os.listdir(data_dir)

    vcnt = 0 # cnt for videos
    for vf in video_folders:
        vcnt += 1
        csv_name = glob.glob(os.path.join(data_dir, vf, '*.csv'))
        df_video = pd.read_csv(csv_name[0])
        print(f'\rProcess: video-{vcnt}/{len(video_folders)}', end='    ')
        all_obj = df_video['ID'].unique()
        
        for i in range(len(all_obj)):
            obj_id = all_obj[i]
            df_obj = df_video[df_video['ID'] == obj_id]
            for T in range(minT,maxT+1):
                sample_list = []
                for i in range(len(df_obj)-past*period-T): # each sample
                    sample = []
                    ################## Sample START ##################
                    for j in range(past+1):
                        time_step = df_obj.iloc[i+j*period]['t']
                        this_x = df_obj.iloc[i+j*period]['x']
                        this_y = df_obj.iloc[i+j*period]['y']
                        obj_info = f'{time_step}_{this_x}_{this_y}'
                        sample.append(obj_info)
                    sample.append(obj_id)
                    sample.append(df_obj.iloc[i+past+T]['x'])
                    sample.append(df_obj.iloc[i+past+T]['y'])
                    sample.append(T)
                    sample.append(df_obj.iloc[i+past+T]['index'])
                    ################## Sample E N D ##################
                    sample_list.append(sample)
                df_T = pd.DataFrame(sample_list, columns=df_all.columns)
                df_all = pd.concat([df_all, df_T], ignore_index=True)
    df_all.to_csv(os.path.join(save_dir, 'all_data.csv'), index=False)

def save_SDD_data(video_reader, save_path: str, period=1, resize_label=True) -> None:    # save as csv file
    # scenario indices: 0~7 (bookstore, coupa, deathCircle, gates, hyang,  little, nexus,  quad)
    # video indices:         0:0~6,     1:0~3, 2:0~4,       3:0~8, 4:0~14, 5:0~3,  6:0~11, 7:0~3
    #                        1          1      1            2      2       1       2       0
    scene_idx = video_reader.scenario_idx
    video_idx = video_reader.video_idx

    id_list = []
    t_list = []   # time or time step
    x_list = []   # x coordinate
    y_list = []   # y coordinate
    idx_list = [] # more information (e.g. scene and video index)

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
            frame = cv2.resize(frame, new_size)
            ### XXX TEST
            # if cnt>1000: break
        except:
            break

        ### Screening conditions
        if df_frame.shape[0] < 1:
            continue
        # df_frame = df_frame.loc[df_frame['label']=='Pedestrian'] # only pedestrians!!!

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

            id_list.append(df_obj["ID"])
            t_list.append(cnt)
            x_list.append(x)
            y_list.append(y)
            idx_list.append(f'{scene_idx}_{video_idx}')

        if save_path is None:
            cv2.imshow("frame", frame)
            cv2.waitKey()
        else:
            folder = os.path.join(save_path,f'{scene_idx}_{video_idx}/')
            Path(folder).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(os.path.join(folder,f'{cnt}.jpg'), frame)

        df = pd.DataFrame({'t':t_list, 'ID':id_list, 'x':x_list, 'y':y_list, 'index':idx_list}).sort_values(by='t', ignore_index=True)
        df.to_csv(os.path.join(save_path, f'{scene_idx}_{video_idx}', f'{original_size[0]}_{original_size[1]}.csv'), index=False)
            
    video_reader.cap.release()
    print()

def save_SDD_data_part(video_reader, save_path: str, test_split: float, period=1, resize_label=True) -> None:    # save as csv file
    # scenario indices: 0~7 (bookstore, coupa, deathCircle, gates, hyang,  little, nexus,  quad)
    # video indices:         0:0~6,     1:0~3, 2:0~4,       3:0~8, 4:0~14, 5:0~3,  6:0~11, 7:0~3
    #                        1          1      1            2      2       1       2       0
    assert(0<test_split<1),('Split ratio must be in (0,1).')
    scene_idx = video_reader.scenario_idx
    video_idx = video_reader.video_idx

    ### Split into training and testing sets
    all_id = np.unique(video_reader.df_data['ID'].values)
    np.random.seed(0)
    np.random.shuffle(all_id)
    id_for_training, id_for_testing = np.split(all_id, [int((len(all_id)-1)*(1-test_split))])

    id_list_training = []
    t_list_training = []   # time or time step
    x_list_training = []   # x coordinate
    y_list_training = []   # y coordinate
    idx_list_training = [] # more information (e.g. scene and video index)

    id_list_testing = []
    t_list_testing = []   # time or time step
    x_list_testing = []   # x coordinate
    y_list_testing = []   # y coordinate
    idx_list_testing = [] # more information (e.g. scene and video index)

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
            ### XXX TEST
            # if cnt>1000: break
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
                id_list_training.append(df_obj["ID"])
                t_list_training.append(cnt)
                x_list_training.append(x)
                y_list_training.append(y)
                idx_list_training.append(f'{scene_idx}_{video_idx}')
            elif df_obj["ID"] in id_for_testing:
                no_testing = False
                id_list_testing.append(df_obj["ID"])
                t_list_testing.append(cnt)
                x_list_testing.append(x)
                y_list_testing.append(y)
                idx_list_testing.append(f'{scene_idx}_{video_idx}')
            else:
                continue

        if save_path is None:
            cv2.imshow("frame", frame)
            cv2.waitKey()
        else:
            if not no_training:
                folder = os.path.join(save_path+'_training',f'{scene_idx}_{video_idx}/')
                Path(folder).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(os.path.join(folder,f'{cnt}.jpg'), frame)
            if not no_testing:
                folder = os.path.join(save_path+'_testing',f'{scene_idx}_{video_idx}/')
                Path(folder).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(os.path.join(folder,f'{cnt}.jpg'), frame)

        if t_list_training:
            df_training = pd.DataFrame({'t':t_list_training, 'ID':id_list_training, 'x':x_list_training, 'y':y_list_training, 'index':idx_list_training}).sort_values(by='t', ignore_index=True)
            df_training.to_csv(os.path.join(save_path+'_training', f'{scene_idx}_{video_idx}', f'{original_size[0]}_{original_size[1]}.csv'), index=False)

        if t_list_testing:
            df_testing = pd.DataFrame({'t':t_list_testing, 'ID':id_list_testing, 'x':x_list_testing, 'y':y_list_testing, 'index':idx_list_testing}).sort_values(by='t', ignore_index=True)
            df_testing.to_csv(os.path.join(save_path+'_testing', f'{scene_idx}_{video_idx}', f'{original_size[0]}_{original_size[1]}.csv'), index=False)

    video_reader.cap.release()
    print()


