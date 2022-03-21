import os
from pathlib import Path

import numpy as np

import torchvision
from data_handle import data_handler, dataset

import matplotlib.pyplot as plt

root_dir = os.path.join(Path(__file__).parents[1], 'Data/SDD_3FPS_Trial/')
composed = torchvision.transforms.Compose([data_handler.ToTensor()])
sdd = dataset.ImageStackDatasetSDD(csv_path=root_dir+'all_data.csv', root_dir=root_dir, transform=composed)

# dh = DataHandler(sdd, batch_size=2, shuffle=True, validation_prop=0.2, validation_cache=None)
# image, label = dh.return_batch()
# print(image.size(), label)

fig, [ax1, ax2] = plt.subplots(1,2)
for i in range(200,3000):
    image = sdd[i]['image']
    traj  = np.array(sdd[i]['traj'])
    label = sdd[i]['label']

    [ax.cla() for ax in [ax1, ax2]]
    ax1.imshow(image[-2,:,:], cmap='gray')
    ax2.imshow(image[-1,:,:], cmap='gray')
    ax1.plot(traj[:,0], traj[:,1], 'rx')
    ax1.plot(label[0], label[1], 'bo')
    plt.pause(0.1)
    input()
plt.show()