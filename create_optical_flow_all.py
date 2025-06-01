from os import path
import os
import numpy as np
import cv2
import time

import pandas
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from torchsummary import summary
from torch.utils.data import TensorDataset, DataLoader
import argparse
from distutils.util import strtobool
import torch
from vit_pytorch import SimpleViT
from vit_pytorch.crossformer import CrossFormer
from Model import HTNet
# from facenest import Fusionmodel
import numpy as np
import cv2 as cv
from facenet_pytorch import MTCNN
from PIL import Image
import pandas as pd


def pol2cart(rho, phi): #Convert polar coordinates to cartesian coordinates for computation of optical strain
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)

def computeStrain(u, v):
    u_x= u - pd.DataFrame(u).shift(-1, axis=1)
    v_y= v - pd.DataFrame(v).shift(-1, axis=0)
    u_y= u - pd.DataFrame(u).shift(-1, axis=0)
    v_x= v - pd.DataFrame(v).shift(-1, axis=1)
    os1 = np.array(np.sqrt(u_x**2 + v_y**2 + 1/2 * (u_y+v_x)**2).ffill(axis=1).ffill(axis=0))
    return os1

def resize(img, target_size=(28,28)):
    rimg = cv2.resize(img, (*target_size,), interpolation=cv2.INTER_LINEAR)
    return rimg

def perChannelNormalize(img):
    res = img.copy()
    for chnl in range(img.shape[2]):
        channel = res[:,:,chnl]
        res[:,:,chnl] = 255.*(channel - channel.min())/(channel.max() - channel.min())
    return res.astype(np.uint8)


def get_whole_u_v_os(df):

    m, n = df.shape
    micro_data_src = 'datasets/CASME3/Part_A_ME_clip/frame'
    macro_data_src = 'datasets/CASME3/Part_A_MaE_clip/frame'

    base_whole_destination_folder = 'datasets/CASME3/casme3_original+mtcnn_tvl1_norm_u_v_os_all_28'
    total_emotion=0
    image_size_u_v = 28
    whole_u_v_os_images = []

    mtcnn = MTCNN(margin=0, image_size=400, select_largest=True, post_process=False, device='cuda:0')
    for i in range(0, m):
        if df['Type'][i] == 'ME':
            base_data_src = micro_data_src
        else:
            base_data_src = macro_data_src

        image_folder = os.path.join(base_data_src, f"{df['Subject'][i]}_{df['Filename'][i]}_{df['Onset'][i]}")
        img_path_apex = os.path.join(image_folder, str(df['Apex'][i])+ '.jpg')
        img_path_onset = os.path.join(image_folder, str(df['Onset'][i])+ '.jpg')
        frames = os.listdir(image_folder)
        frames = sorted(frames, key=lambda x: int(x.split('.')[0]))
        num = len(frames)
        assert len(frames) > 0, f"Folder {image_folder} is empty"

        if not os.path.exists(img_path_apex):
            img_path_apex = os.path.join(image_folder, frames[num//2])
            print(f"Apex {img_path_apex} does not exist, using {img_path_apex} instead")

        if not os.path.exists(img_path_onset):
            img_path_onset = os.path.join(image_folder, frames[0])
            print(f"Onset {img_path_onset} does not exist, using {img_path_onset} instead")

        if img_path_apex == img_path_onset:
            img_path_apex = os.path.join(image_folder, frames[num//2])
            if img_path_apex == img_path_onset:
                img_path_apex = os.path.join(image_folder, frames[1])
            print(f"Apexand Onset are the same {img_path_onset} , using {img_path_apex} instead")

        train_face_image_apex = cv2.imread(img_path_apex)
        train_face_image_apex = cv2.cvtColor(train_face_image_apex, cv2.COLOR_BGR2RGB)
        train_face_image_apex = Image.fromarray(train_face_image_apex)

        train_face_image_onset = cv2.imread(img_path_onset)
        train_face_image_onset = cv2.cvtColor(train_face_image_onset, cv2.COLOR_BGR2RGB)
        train_face_image_onset = Image.fromarray(train_face_image_onset)
        # get face and bounding box
        side = max(train_face_image_apex.size[0], train_face_image_apex.size[1])
        if train_face_image_apex.size == train_face_image_onset.size:
            if train_face_image_apex == train_face_image_onset:
                print(f"Image is the same for {img_path_onset} and {img_path_apex}, using {os.path.join(image_folder, frames[-1])} instead")
                img_path_apex = os.path.join(image_folder, frames[-1])
                train_face_image_apex = cv2.imread(img_path_apex)
                train_face_image_apex = cv2.cvtColor(train_face_image_apex, cv2.COLOR_BGR2RGB)
                train_face_image_apex = Image.fromarray(train_face_image_apex)
                if train_face_image_apex == train_face_image_onset:
                    print(f"Image is the same for {img_path_onset} and {img_path_apex}")
                    whole_u_v_os_images.append(None)
                    continue
        box1, score1, points1 = mtcnn.detect(train_face_image_onset, landmarks=True)
        box2, score2, points2 = mtcnn.detect(train_face_image_apex, landmarks=True)
        if box1 is None or box2 is None:
            print(f"Face is None for {img_path_onset} or {img_path_apex}")
            whole_u_v_os_images.append(None)
            continue
        box1, box2 = box1[0].astype(int), box2[0].astype(int)
        points1, points2 = points1[0], points2[0]
        eye_dis1 = np.linalg.norm(points1[0] - points1[1])*5/12
        eye_dis2 = np.linalg.norm(points2[0] - points2[1])*5/12
        box1[0] = int(max(0, points1[0][0] - eye_dis1*0.9))
        box1[1] = int(max(0, points1[0][1] - eye_dis1*1.4))
        box1[2] = int(min(train_face_image_onset.size[1], points1[1][0] + eye_dis1*0.9))
        box2[0] = int(max(0, points2[0][0] - eye_dis2*0.9))
        box2[1] = int(max(0, points2[0][1] - eye_dis2*1.4))
        box2[2] = int(min(train_face_image_apex.size[1], points2[1][0] + eye_dis2*0.9))
        train_face_image_onset = train_face_image_onset.crop(box1) 
        train_face_image_apex = train_face_image_apex.crop(box2)
        
        train_face_image_onset = np.array(train_face_image_onset)
        train_face_image_apex = np.array(train_face_image_apex)
        face_onset = cv2.resize(train_face_image_onset, (image_size_u_v, image_size_u_v), interpolation=cv2.INTER_LINEAR)
        face_apex = cv2.resize(train_face_image_apex, (image_size_u_v, image_size_u_v), interpolation=cv2.INTER_LINEAR)

        pre_face_onset = cv2.cvtColor(face_onset, cv2.COLOR_RGB2GRAY)
        next_face_apex = cv2.cvtColor(face_apex, cv2.COLOR_RGB2GRAY)

        flow = cv2.optflow.DualTVL1OpticalFlow_create(
        tau = 0.25, 
        #lambda_ = 0.15,
        theta = 0.3,
        nscales = 5, 
        warps=5,
        epsilon=0.01, 
        innnerIterations=30,
        outerIterations=10, 
        scaleStep=0.5,
        gamma=0.1,
        medianFiltering=5,
        useInitialFlow=False
    )
        flow = flow.calc(pre_face_onset, next_face_apex, None)
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1]) 
        u, v = pol2cart(magnitude, angle)
        os1 = computeStrain(u, v)
                    
        #Features Concatenation
        final = np.zeros((*pre_face_onset.shape[0:2],3))
        final[:,:,0] = os1 #B
        final[:,:,1] = v #G
        final[:,:,2] = u #R
        final = resize(final, target_size=(image_size_u_v, image_size_u_v))
        final = perChannelNormalize(final)
        whole_u_v_os_images.append(final)


        if face_onset is not None:
            total_emotion = total_emotion + 1

        if df['Type'][i] == 'ME': 
            destination_folder = os.path.join(base_whole_destination_folder, 'ME')
        else:
            destination_folder = os.path.join(base_whole_destination_folder, 'MaE')
        os.makedirs(destination_folder, exist_ok=True)
        saved_filename = str(df['Subject'][i]) + '_'+str(df['Filename'][i])+'_'+str(df['Apex'][i])+'.png'
        if os.path.exists(os.path.join(destination_folder, saved_filename)):
            continue
        cv2.imwrite(os.path.join(destination_folder, saved_filename), final)

    print(np.shape(whole_u_v_os_images))
    print(total_emotion)

    return whole_u_v_os_images

def create_norm_u_v_os_train_test():
    df_micro = pandas.read_excel('datasets/CASME3/annotation/cas(me)3_part_A_ME_label_JpgIndex_v2_final.xlsx')
    df_micro['Type'] = 'ME'
    df_macro = pandas.read_excel('datasets/CASME3/annotation/cas(me)3_part_A_MaE_label_JpgIndex_v2_emotion.xlsx')
    df_macro['Type'] = 'MaE'

    micro_num = df_micro.shape[0]
    macro_num = df_macro.shape[0]
    df = pandas.concat([df_micro, df_macro], ignore_index=True)
    m, n = df.shape
    whole_u_v_os_Arr = get_whole_u_v_os(df)

    print('finish get')


if __name__ == '__main__':
    create_norm_u_v_os_train_test()