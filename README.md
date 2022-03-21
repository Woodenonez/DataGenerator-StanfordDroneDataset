# Data Generator of Stanford Drone Dataset for Motion Prediction
This is used to generate datasets from raw videos of the Stanford Drone Dataset (SDD).

# Scripts
1. Use "check_all_traj" to check information and trajectories in videos and scenes.
2. Use "gen_dataset" to generate datasets.

# Generate datasets
In "gen_dataset", there are two modes controlled by the parameter "test_split":
1. If "test_split==0", selected videos will be converted into a dataset with no split;
2. If "test_split>0",  selected videos will be split into training and testing subsets with a ratio of "test_split".
(No automatic parameter check, make sure "test_split" belongs to [0,1).)

Other parameters:
- "data_dir" is the root directory of the raw SDD.
- "past" is the number of past frames you want to include as inputs.
- "img_saving_period" is the periodof saving frames from video stream, which is used to control/change the FPS.
- "dataset_gen_period" is not important, remain it to 1.
- "minT" and "maxT" are minimal and maximal prediction time offsets.
- "all_video_idx" is all videos that are meant to be converted into datasets. 

# Dataset structure
Data - dataset folder - scence_video folders + dataset csv - frames + info csv
