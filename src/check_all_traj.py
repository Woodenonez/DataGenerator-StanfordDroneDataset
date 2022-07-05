import numpy as np
import matplotlib.pyplot as plt

from load_video import ReadVideo

'''
# scenario indices: 0~7 (bookstore, coupa, deathCircle, gates, hyang,  little, nexus,  quad)
# video indices:         0:0~6,     1:0~3, 2:0~4,       3:0~8, 4:0~14, 5:0~3,  6:0~11, 7:0~3
#                        1          1      1            2      2       1       2       0
# label: Biker, Pedestrian, Skater, Cart, Car, Bus
'''

def check_SDD(data_dir, scenario_list, video_list, vis=False):
    for scenario_name in scenario_list:
        for video_name in video_list:
            print(f'Check scenario {scenario_name}-{video_name}.')

            video_reader = ReadVideo(data_dir, scenario_name, video_name, verbose=False)
            duration = int(video_reader.info('nframes'))/video_reader.info('fps')
            resolution = (int(video_reader.info('width')), int(video_reader.info('height')))
            base_map, _ = video_reader.read_frame_clean(after=200)
            video_reader.cap.release()

            df = video_reader.df_data
            id_list = df['ID'].unique()
            label_list = df['label'].unique()
            label_msg = ''
            for label in label_list:
                df_this_label = df.loc[df["label"]==label]
                this_label_msg = f'{label}-{len(df_this_label["ID"].unique())}'
                label_msg = label_msg + this_label_msg + ', '
            print(f'Stats ({int(duration//60)}:{int(duration%60)}, WxH{resolution}): #objects={len(id_list)} ({label_msg})')

            if vis:
                _, ax = plt.subplots()
                ax.imshow(base_map)
                for id in id_list:
                    df_id = df.loc[df['ID']==id]
                    if df_id['label'].iloc[0] != 'Pedestrian':
                        continue
                    x_id = (df_id['xmin']+df_id['xmax'])/2
                    y_id = (df_id['ymin']+df_id['ymax'])/2
                    traj_id = np.vstack((x_id.to_numpy(), y_id.to_numpy()))
                    ax.plot(traj_id[0,:],traj_id[1,:],'.', markersize=1)
                    # plt.pause(0.1)
                plt.show()


if __name__ == '__main__':
    data_dir = '/media/ze/Elements/User/Data/SDD/'

    # scenario_list = ['bookstore', 'coupa', 'deathCircle', 'gates', 'hyang', 'little', 'nexus', 'quad']
    # video_list = [f'video{i}' for i in range(15)]

    scenario_list = ['hyang']
    video_list = [f'video{i}' for i in range(1)]

    check_SDD(data_dir, scenario_list, video_list, vis=True)