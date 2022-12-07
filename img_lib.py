import cv2
import cvxpy
import os

def getimg(file_name):
    return cv2.imread(file_name,0)

def getimg_downsample(file_name,n):
    img = cv2.imread(file_name, 0)
    return cv2.resize(img, (int(img.shape[0]/n), int(img.shape[1]/n)))


def getimg_dataset(folder_name, downsample_rate=1, size = 10):
    file_name_list = os.listdir(folder_name)[:size]
    print(file_name_list)
    return [(name[:-4], getimg_downsample(folder_name+"\\"+name, downsample_rate)) for name in file_name_list]
