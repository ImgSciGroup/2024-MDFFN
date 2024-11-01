'''
 Dataset Source:
    Farmland: http://crabwq.github.io/
    River: https://share.weiyun.com/5ugrczK
    Hermiston: https://citius.usc.es/investigacion/datasets/hyperspectral-change-detection-dataset
'''
import os

from PIL import Image



from scipy.io import loadmat
import numpy as np
import  torch
from sklearn.preprocessing import MinMaxScaler
def get_fire_dataset():
    data_set_before = np.array(Image.open(r'../datasets/fire/pre.tif'))
    data_set_after = np.array(Image.open(r'../datasets/fire/pre.tif'))
    ground_truth=np.array(Image.open(r'../datasets/fire/GT.bmp'))
    ground_truth=ground_truth[:,:,0]
    ground_truth = ground_truth.flatten()
    for i in range(len(ground_truth)):
        if ground_truth[i] == 0:
            ground_truth[i] = 0
        elif ground_truth[i] != 0:
            ground_truth[i] = 1
    ground_truth = ground_truth.reshape((data_set_before.shape[0], data_set_before.shape[1]))

    img1 = data_set_before.astype('float32')
    img2 = data_set_after.astype('float32')
    gt = ground_truth.astype('float32')

    return img1, img2, gt
def get_Wetland_dataset():
    data_set_before = loadmat(r'D:\PyProject\ChangeDetection-MSDFFN-master\datasets\Wetland_new\PreImg_2006_new.mat')['img']
    data_set_after = loadmat(r'D:\PyProject\ChangeDetection-MSDFFN-master\datasets\Wetland_new\PostImg_2007_new.mat')['img']
    ground_truth = loadmat(r'D:\PyProject\ChangeDetection-MSDFFN-master\datasets\Wetland_new\Reference_Map_Binary_new.mat')['new_Wetland_gt']
    ground_truth = np.array(ground_truth).flatten()
    for i in range(len(ground_truth)):
        if ground_truth[i] == 0:
            ground_truth[i] = 0
        elif ground_truth[i] != 0:
            ground_truth[i] = 1
    ground_truth = ground_truth.reshape((data_set_before.shape[0], data_set_before.shape[1]))

    img1 = data_set_before.astype('float32')
    img2 = data_set_after.astype('float32')
    gt = ground_truth.astype('float32')

    return img1, img2, gt

def get_Wetland_new_dataset():
    data_set_before = loadmat(r'../datasets/Yancheng/farm06.mat')['imgh']
    data_set_after = loadmat(r'../datasets/Yancheng/farm07.mat')['imghl']
    ground_truth = loadmat(r'../datasets/Yancheng/label.mat')['label']
    ground_truth = np.array(ground_truth).flatten()
    for i in range(len(ground_truth)):
        if ground_truth[i] == 0:
            ground_truth[i] = 0
        elif ground_truth[i] != 0:
            ground_truth[i] = 1
    ground_truth = ground_truth.reshape((data_set_before.shape[0], data_set_before.shape[1]))

    img1 = data_set_before.astype('float32')
    img2 = data_set_after.astype('float32')
    gt = ground_truth.astype('float32')

    return img1, img2, gt


def get_Hermiston_dataset():
    data_set_before = loadmat(r'../datasets/Hermiston/hermiston2004.mat')['HypeRvieW']
    data_set_after = loadmat(r'../datasets/Hermiston/hermiston2007.mat')['HypeRvieW']
    ground_truth = loadmat(r'../datasets/Hermiston/rdChangesHermiston_5classes.mat')['gt5clasesHermiston']
    ground_truth = np.array(ground_truth).flatten()
    for i in range(len(ground_truth)):
        if ground_truth[i] == 0:
            ground_truth[i] = 0
        elif ground_truth[i] != 0:
            ground_truth[i] = 1
    ground_truth = ground_truth.reshape((data_set_before.shape[0], data_set_before.shape[1]))

    img1 = data_set_before.astype('float32')
    img2 = data_set_after.astype('float32')
    gt = ground_truth.astype('float32')

    return img1, img2, gt


def get_river_dataset():
    data_set_before = loadmat(r'../datasets/River/river_before.mat')['river_before']
    data_set_after = loadmat(r'../datasets/River/river_after.mat')['river_after']
    ground_truth = loadmat(r'../datasets/River/groundtruth.mat')['lakelabel_v1']
    ground_truth = np.array(ground_truth).flatten()
    for i in range(len(ground_truth)):
        if ground_truth[i] == 0:
            ground_truth[i] = 0
        elif ground_truth[i] != 0:
            ground_truth[i] = 1
    ground_truth = ground_truth.reshape((data_set_before.shape[0], data_set_before.shape[1]))

    img1 = data_set_before.astype('float32')
    img2 = data_set_after.astype('float32')
    gt = ground_truth.astype('float32')

    return img1, img2, gt

def get_BayArea_dataset():
    data_set_before = loadmat(r'../datasets/BayArea/Bay_Area_2013.mat')['HypeRvieW']
    data_set_after = loadmat(r'../datasets/BayArea/Bay_Area_2015.mat')['HypeRvieW']
    ground_truth = loadmat(r'../datasets/BayArea/bayArea_gtChangesolf.mat')['HypeRvieW']
    ground_truth = np.array(ground_truth).flatten()
    for i in range(len(ground_truth)):
        if ground_truth[i] == 0:
            ground_truth[i] = 0
        elif ground_truth[i] != 0:
            ground_truth[i] = 1
    ground_truth = ground_truth.reshape((data_set_before.shape[0], data_set_before.shape[1]))

    img1 = data_set_before.astype('float32')
    img2 = data_set_after.astype('float32')
    gt = ground_truth.astype('float32')

    return img1, img2, gt

def get_Santa_dataset():
    data_set_before = loadmat(r'../datasets/Santa/barbara_2013.mat')['HypeRvieW']
    data_set_after = loadmat(r'../datasets/Santa/barbara_2014.mat')['HypeRvieW']
    ground_truth = loadmat(r'../datasets/Santa/barbara_gtChanges.mat')['HypeRvieW']

    # scaler = MinMaxScaler(feature_range=(0, 1))
    # for i in range(data_set_before.shape[0]):
    #     data_set_before[i, :, :] = scaler.fit_transform(data_set_before[i, :, :])
    #     data_set_after[i, :, :] = scaler.fit_transform(data_set_after[i, :, :])
    ground_truth = np.array(ground_truth).flatten()
    for i in range(len(ground_truth)):
        if ground_truth[i] == 0:
            ground_truth[i] = 0
        elif ground_truth[i] != 0:
            ground_truth[i] = 1
    ground_truth = ground_truth.reshape((data_set_before.shape[0], data_set_before.shape[1]))

    img1 = data_set_before.astype('float32')
    img2 = data_set_after.astype('float32')
    gt = ground_truth.astype('float32')

    return img1, img2, gt

def get_island_dataset():
    data_set_before = np.array(Image.open(r'..\datasets/8_Island town/Img11-Ac.png'))
    data_set_after = np.array(Image.open(r'..\datasets/8_Island town/Img11-B.png'))
    ground_truth = np.array(Image.open(r'..\datasets/8_Island town/gt.png'))
    ground_truth = ground_truth[:, :, 0]
    ground_truth = ground_truth.flatten()
    for i in range(len(ground_truth)):
        if ground_truth[i] == 0:
            ground_truth[i] = 0
        elif ground_truth[i] != 0:
            ground_truth[i] = 1
    ground_truth = ground_truth.reshape((data_set_before.shape[0], data_set_before.shape[1]))

    img1 = data_set_before.astype('float32')
    img2 = data_set_after.astype('float32')
    gt = ground_truth.astype('float32')

    return img1, img2, gt
def get_B_dataset():
    data_set_before = np.array(Image.open(r'..\datasets/B/1.tif'))
    data_set_after = np.array(Image.open(r'..\datasets/B/2.tif'))
    ground_truth = np.array(Image.open(r'..\datasets/B/3.tif'))
   # ground_truth = ground_truth[:, :, 0]
    ground_truth = ground_truth.flatten()
    for i in range(len(ground_truth)):
        if ground_truth[i] == 0:
            ground_truth[i] = 0
        elif ground_truth[i] != 0:
            ground_truth[i] = 1
    ground_truth = ground_truth.reshape((data_set_before.shape[0], data_set_before.shape[1]))

    img1 = data_set_before.astype('float32')
    img2 = data_set_after.astype('float32')
    gt = ground_truth.astype('float32')

    return img1, img2, gt

def get_HK_dataset():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_set_before = np.array(Image.open(r'../datasets/HK/1.png'))
    data_set_after = np.array(Image.open(r'../datasets/HK/2.png'))
    ground_truth = np.array(Image.open(r'../datasets/HK/3.png'))
    ground_truth = ground_truth[:, :, 0]
    ground_truth = ground_truth.flatten()
    for i in range(len(ground_truth)):
        if ground_truth[i] == 0:
            ground_truth[i] = 0
        elif ground_truth[i] != 0:
            ground_truth[i] = 1
    ground_truth = ground_truth.reshape((data_set_before.shape[0], data_set_before.shape[1]))

    img1 = data_set_before.astype('float32')
    img2 = data_set_after.astype('float32')
    gt = ground_truth.astype('float32')

    return img1, img2, gt
def get_dataset(current_dataset):
    if current_dataset == 'Farmland':
        return get_Wetland_new_dataset()   # Farmland(450, 140, 155), gt[0. 1.]

    if current_dataset == 'Hermiston':
        return get_Hermiston_dataset()   #
    if current_dataset == 'River':
        return get_river_dataset()
    if current_dataset == 'BayArea':
        return get_BayArea_dataset()
    if current_dataset == 'Santa':
        return get_Santa_dataset()
    if current_dataset == 'Wetland':
        return get_Wetland_dataset()
    if current_dataset == 'fire':
        return get_fire_dataset()
    if current_dataset == 'island':
        return get_island_dataset()
    if current_dataset == 'B':
        return get_B_dataset()
    if current_dataset == 'HK':
        return get_HK_dataset()


