import os
import json
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import cv2
from facenet_pytorch import MTCNN

def whole_face_block_coordinates(base_data_src = './datasets/combined_datasets_whole'):
    # df = pd.read_csv('combined_3_class2_for_optical_flow.csv')
    # m, n = df.shape
    base_data_src = base_data_src
    emotion_imgs = os.listdir(base_data_src)
    total_emotion = len(emotion_imgs)
    image_size_u_v = 28
    # get the block center coordinates
    face_block_coordinates = {}

    # for i in range(0, m):
    for image_name in emotion_imgs:
        # image_name = str(df['sub'][i]) + '_' + str(
        #     df['filename_o'][i]) + ' .png'
        # print(image_name)
        # img_path_apex = base_data_src + '/' + df['imagename'][i]
        img_path_apex = base_data_src + '/' + image_name
        image_name = image_name.split('.jpg')[0].strip()
        train_face_image_apex = cv2.imread(img_path_apex) # (444, 533, 3)
        face_apex = cv2.resize(train_face_image_apex, (28,28), interpolation=cv2.INTER_AREA)
        # get face and bounding box
        mtcnn = MTCNN(margin=0, image_size=image_size_u_v, select_largest=True, post_process=False, device='cuda:0')
        batch_boxes, _, batch_landmarks = mtcnn.detect(face_apex, landmarks=True)
        # print(img_path_apex,batch_landmarks)
        # if not detecting face
        batch_landmarks = None
        if batch_landmarks is None:
            # print(f"landmarks is None for {img_path_apex}")
            batch_landmarks = np.array([[[9.528073, 11.062551]
                                            , [21.396168, 10.919773]
                                            , [15.380184, 17.380562]
                                            , [10.255435, 22.121233]
                                            , [20.583706, 22.25584]]])
            # batch_landmarks = np.array([[[7, 7]
            #                                 , [21, 7]
            #                                 , [14, 14]
            #                                 , [7, 21]
            #                                 , [21, 21]]])
            # print(img_path_apex)
        else:
            print(f"landmarks is not None for {img_path_apex}")
            # import ipdb; ipdb.set_trace()
        row_n, col_n = np.shape(batch_landmarks[0])
        # print(batch_landmarks[0])
        for i in range(0, row_n):
            for j in range(0, col_n):
                if batch_landmarks[0][i][j] < 7:
                    batch_landmarks[0][i][j] = 7
                if batch_landmarks[0][i][j] > 21:
                    batch_landmarks[0][i][j] = 21
        batch_landmarks = batch_landmarks.astype(int)
        # print(batch_landmarks[0])
        # get the block center coordinates
        face_block_coordinates[image_name] = batch_landmarks[0]
    # print(len(face_block_coordinates))
    return face_block_coordinates

def crop_optical_flow_block(data_type="casme3"):
    
    if data_type == "casme2":
        base_data_src, optical_flow_path = 'datasets/combined_datasets_whole', 'datasets/STSNet_whole_norm_u_v_os'
    elif data_type == "casme3":
        base_data_src, optical_flow_path = 'datasets/casme3_crop_datasets_whole', 'datasets/casme3_crop_tvl1_whole_norm_u_v_os'
    
    face_block_coordinates_dict = whole_face_block_coordinates(base_data_src=base_data_src)
    # print(len(face_block_coordinates_dict))
    # Get train dataset
    whole_optical_flow_path = optical_flow_path
    whole_optical_flow_imgs = os.listdir(whole_optical_flow_path)
    four_parts_optical_flow_imgs = {}
    # print(whole_optical_flow_imgs[0]) #spNO.189_f_150.png
    for n_img in whole_optical_flow_imgs:
        four_parts_optical_flow_imgs[n_img]=[]
        flow_image = cv2.imread(whole_optical_flow_path + '/' + n_img)
        if data_type == 'casme2':
            if not n_img.startswith('sub'):
                continue
        four_part_coordinates = face_block_coordinates_dict[n_img.split('.png')[0].strip()]
        l_eye = flow_image[four_part_coordinates[0][0]-7:four_part_coordinates[0][0]+7,
                four_part_coordinates[0][1]-7: four_part_coordinates[0][1]+7]
        l_lips = flow_image[four_part_coordinates[1][0] - 7:four_part_coordinates[1][0] + 7,
                four_part_coordinates[1][1] - 7: four_part_coordinates[1][1] + 7]
        nose = flow_image[four_part_coordinates[2][0] - 7:four_part_coordinates[2][0] + 7,
                four_part_coordinates[2][1] - 7: four_part_coordinates[2][1] + 7]
        r_eye = flow_image[four_part_coordinates[3][0] - 7:four_part_coordinates[3][0] + 7,
                four_part_coordinates[3][1] - 7: four_part_coordinates[3][1] + 7]
        r_lips = flow_image[four_part_coordinates[4][0] - 7:four_part_coordinates[4][0] + 7,
                four_part_coordinates[4][1] - 7: four_part_coordinates[4][1] + 7]
        four_parts_optical_flow_imgs[n_img].append(l_eye)
        four_parts_optical_flow_imgs[n_img].append(l_lips)
        four_parts_optical_flow_imgs[n_img].append(nose)
        four_parts_optical_flow_imgs[n_img].append(r_eye)
        four_parts_optical_flow_imgs[n_img].append(r_lips)
    # import ipdb; ipdb.set_trace()
        # print(np.shape(l_eye))
    # print((four_parts_optical_flow_imgs['spNO.189_f_150.png'][0]))->(14,14,3)
    print(len(four_parts_optical_flow_imgs))
    return four_parts_optical_flow_imgs


def add_noise(au_sequence, noise_level=0.05):
    """
    Add random noise to AU sequence.
    """
    noise = np.random.uniform(-noise_level, noise_level, size=au_sequence.shape)
    au_sequence_noisy = au_sequence + noise
    # Clip the values to remain in the 0-1 range
    au_sequence_noisy = np.clip(au_sequence_noisy, 0.0, 1.0)
    return au_sequence_noisy

def scale_au(au_sequence, scale_factor_range=(0.95, 1.05)):
    """
    Scale the AU sequence by a random factor.
    """
    scale_factor = np.random.uniform(scale_factor_range[0], scale_factor_range[1])
    au_sequence_scaled = au_sequence * scale_factor
    # Clip the values to remain in the 0-1 range
    au_sequence_scaled = np.clip(au_sequence_scaled, 0.0, 1.0)
    return au_sequence_scaled

def shift_au(au_sequence, shift_range=(-0.05, 0.05)):
    """
    Shift the AU sequence by a random amount.
    """
    shift = np.random.uniform(shift_range[0], shift_range[1])
    au_sequence_shifted = au_sequence + shift
    # Clip the values to remain in the 0-1 range
    au_sequence_shifted = np.clip(au_sequence_shifted, 0.0, 1.0)
    return au_sequence_shifted

def dropout_au(au_sequence, dropout_rate=0.1):
    """
    Apply dropout to the AU sequence.
    """
    mask = np.random.choice([0, 1], size=au_sequence.shape, p=[dropout_rate, 1 - dropout_rate])
    au_sequence_dropped = au_sequence * mask
    return au_sequence_dropped

def interpolate_au(au_sequence, interpolation_factor=0.1):
    """
    Interpolate the AU sequence by adding a small random variation.
    """
    interpolation = np.random.uniform(-interpolation_factor, interpolation_factor, size=au_sequence.shape)
    au_sequence_interpolated = au_sequence + interpolation
    au_sequence_interpolated = np.clip(au_sequence_interpolated, 0.0, 1.0)
    return au_sequence_interpolated


class CASMEDataset(Dataset):
    def __init__(self, 
                 mode='train', 
                 test_subject='spNO.1', 
                 json_path='datasets/CASME3/ME_inference_au.json', 
                 xlsx_path='datasets/CASME3/annotation/cas(me)3_part_A_ME_label_JpgIndex_v2_refined.xlsx', 
                 data_path='datasets/CASME3/Part_A_ME_clip_scrfd_cropped',
                 optical_flow_path='datasets/three_norm_u_v_os',
                 transform_au=False,
                 num_classes=7,
                 n_frames=32, 
                 pad_value=0.0,
                 data_type='casme3',
                 first_transform=None,
                 img_transform=None):
        """
        CASME3 Dataset for micro-expression classification.

        Args:
            mode (str): Either 'train' or 'test'.
            test_subject (str): The subject to be used for testing.
            json_path (str): Path to the JSON file containing AU values.
            xlsx_path (str): Path to the Excel file containing annotations.
            n_frames (int): Number of frames to return for each sample.
            pad_value (float): Value to use for padding sequences shorter than n_frames.
        """
        self.mode = mode
        self.test_subject = test_subject
        self.n_frames = n_frames
        self.pad_value = pad_value
        self.data_path = data_path
        self.transform_au = transform_au
        self.emotion_dict =  {'others': 0, 'happy': 1, 'disgust': 2, 'sad': 3, 'fear': 4, 'anger': 5, 'surprise': 6}
        self.emotion_dict_casme2 = {'others': 0, 'happiness': 1, 'disgust': 2, 'sadness': 3, 'fear': 4, 'repression': 5, 'surprise': 6}
        self.simple_emotion_dict = {'anger': 0, 'disgust': 0, 'fear': 0, 'happy': 1, 'sad': 0, 'surprise': 2}
        self.simple_emotion_dict_casme2 = {'happiness':1, 'disgust':0, 'sadness':0, 'fear':0, 'repression':0, 'surprise':2, 'others':3}
        self.num_classes = num_classes
        self.data_type = data_type
        self.use_optical_flow =  True
        self.use_img = False
        self.img_transform = img_transform
        self.first_transform = first_transform
        # Load AU data
        with open(json_path, 'r') as f:
            self.au_data = json.load(f)

        # Load annotations
        self.annotations = pd.read_excel(xlsx_path)
        # Filter annotations based on mode
        if self.mode == 'train':
            if data_type == 'casme3':
                self.annotations = self.annotations[self.annotations['Subject'] != self.test_subject]
            elif data_type == 'casme2':
                self.annotations = self.annotations[self.annotations['Subject'] != int(self.test_subject.replace('sub', ''))]
        elif self.mode == 'test':
            if data_type == 'casme3':
                self.annotations = self.annotations[self.annotations['Subject'] == self.test_subject]
            elif data_type == 'casme2':
                self.annotations = self.annotations[self.annotations['Subject'] == int(self.test_subject.replace('sub', ''))]
        # Encode labels
        self.label_encoder = LabelEncoder()
        # lower labels
        if data_type == 'casme3':
            self.annotations['Objective class'] = self.annotations['Objective class'].str.lower()
            self.annotations['emotion'] = self.annotations['emotion'].str.lower()
            self.annotations['Objective class'] = self.label_encoder.fit_transform(self.annotations['Objective class'])
            
        elif data_type == 'casme2':
            # change 'Estimated Emotion' to 'emotion' for casme2
            self.annotations['emotion'] = self.annotations['Estimated Emotion']
            self.annotations['emotion'] = self.annotations['emotion'].str.lower()
            self.annotations['Onset'] = self.annotations['OnsetFrame']
            self.annotations['Offset'] = self.annotations['OffsetFrame']
            self.annotations['Apex'] = self.annotations['ApexFrame']
        
        print(self.annotations['emotion'].value_counts())
        
        # use label dict to encode emotion
        if self.num_classes == 3:
            # remove 'others' class
            self.annotations = self.annotations[self.annotations['emotion'] != 'others']
            if data_type == 'casme2':
                self.annotations = self.annotations[self.annotations['emotion'] != 'fear']
                self.annotations = self.annotations[self.annotations['emotion'] != 'sadness']
                self.annotations = self.annotations[self.annotations['ApexFrame'] != '/'] # sub04_EP12_01f 
                self.annotations = self.annotations[self.annotations['ApexFrame'] != self.annotations['OnsetFrame']] # sub05_EP09_05f

            if data_type == 'casme3':
                # File datasets/CASME3/Part_A_ME_clip/frame/spNO.149_d_112/83.jpg does not exist
                # Face onset is None for datasets/CASME3/Part_A_ME_clip/frame/spNO.184_c_466/466.jpg
                # Face apex is None for datasets/CASME3/Part_A_ME_clip/frame/spNO.2_e_138/143.jpg
                # File datasets/CASME3/Part_A_ME_clip/frame/spNO.216_e_0/0.jpg does not exist
                # Face apex is None for datasets/CASME3/Part_A_ME_clip/frame/spNO.39_j_855/860.jpg
                # Face apex is None for datasets/CASME3/Part_A_ME_clip/frame/spNO.40_j_1016/1023.jpg
                self.annotations = self.annotations[self.annotations['Apex'] != self.annotations['Onset']]
                self.annotations['emotion'] = self.annotations['emotion'].map(self.simple_emotion_dict)
            elif data_type == 'casme2':
                self.annotations['emotion'] = self.annotations['emotion'].map(self.simple_emotion_dict_casme2)
            # self.annotations['emotion'] = self.annotations['emotion'].map(self.simple_emotion_dict)
            print(f'positive samples: {len(self.annotations[self.annotations["emotion"] == 1])}, negative samples: {len(self.annotations[self.annotations["emotion"] == 0])}, surprise samples: {len(self.annotations[self.annotations["emotion"] == 2])}')
            
        else:
            if data_type == 'casme3':
                self.annotations['emotion'] = self.annotations['emotion'].map(self.emotion_dict)
            elif data_type == 'casme2':
                self.annotations['emotion'] = self.annotations['emotion'].map(self.emotion_dict_casme2)
               
        # get optical flow data

        self.all_five_parts_optical_flow = crop_optical_flow_block(self.data_type) # 根据五个部分的坐标，裁剪出四个部分的光流图像

    def __len__(self):
        return len(self.annotations)

    def calculate_label_distribution(self):
        """
        Calculate the distribution of labels in the dataset.
        """
        label_distribution = self.annotations['emotion'].value_counts(normalize=True)
        return label_distribution
    
    def get_class_weights(self):
        """
        Calculate class weights for the dataset.
        """
        label_distribution = self.calculate_label_distribution()
        # import ipdb; ipdb.set_trace()
        class_weights = 1 / label_distribution
        class_weights = class_weights / class_weights.sum()
        class_weights = [class_weights[i] for i in range(self.num_classes)]
        class_weights = torch.tensor(class_weights, dtype=torch.float32)
        return class_weights

    def get_sample_weights(self):
        """
        Calculate sample weights for the dataset.
        """
        label_distribution = self.calculate_label_distribution()
        sample_weights = 1 / label_distribution[self.annotations['emotion']]
        sample_weights = sample_weights / sample_weights.sum()
        sample_weights = sample_weights.values
        return sample_weights

    def __getitem__(self, idx):
        # Get annotation row
        row = self.annotations.iloc[idx]
        subject = row['Subject']
        filename = row['Filename']
        onset = row['Onset']
        apex = row['Apex']
        offset = row['Offset']
        # objective_class = row['Objective class']
        emotion = row['emotion']

        # Construct frame prefix
        frame_prefix = f"{subject}_{filename}_{onset}"

        # Collect AU values for frames from onset to offset
        au_sequence = []
        idx = 0
        apex_idx = -1
        for frame_num in range(onset, offset + 1):
            if self.data_type == 'casme3':
                frame_name =  os.path.join(frame_prefix, f"{frame_num}.jpg")
            elif self.data_type == 'casme2':
                frame_name =  os.path.join('sub'+str(subject).zfill(2), filename, f"img{frame_num}.jpg")
            # print(os.path.join(self.data_path, frame_name))
            if os.path.exists(os.path.join(self.data_path, frame_name)):
                assert frame_name in self.au_data.keys(), f"Frame {frame_name} not found in AU data."
                au_sequence.append(self.au_data[frame_name])
                if frame_num == apex:
                    apex_idx = idx
                idx += 1
        if apex_idx == -1:
            apex_idx = len(au_sequence) // 2
        # 以apex为中心，取前后各16帧
        # if len(au_sequence) > self.n_frames:
        if apex_idx < self.n_frames // 2:  # If apex is too close to the beginning
            au_sequence = [au_sequence[0]] * (self.n_frames // 2 - apex_idx) + au_sequence
            apex_idx = self.n_frames // 2
        if len(au_sequence) - (apex_idx + 1) < self.n_frames // 2:  # If apex is too close to the end
            au_sequence = au_sequence + [au_sequence[-1]] * (self.n_frames // 2 - (len(au_sequence) - (apex_idx + 1)))
        if len(au_sequence) > self.n_frames: # If there are more frames than needed
            start = apex_idx - self.n_frames // 2
            end = apex_idx + self.n_frames // 2 + 1 
            au_sequence = au_sequence[start:end]

        if len(au_sequence) == 0 or len(au_sequence) > 100:
            if len(au_sequence) > 0:
                print(f"more than 100 images in AU sequence for {frame_prefix}. Fetching a random sample.")
            else:
                print(f"Empty AU sequence for {frame_prefix}. Fetching a random sample.")
            # Randomly choose another index to fetch a non-empty sample
            if self.mode == 'train':
                random_idx = random.randint(0, len(self.annotations) - 1)
                return self.__getitem__(random_idx)
            else:
                return self.__getitem__(0)

        au_sequence = np.array(au_sequence, dtype=np.float32)  # Convert to numpy array

        if self.transform_au and self.mode == 'train':
            au_sequence = self.augment_au(au_sequence)

        au_sequence = torch.tensor(au_sequence, dtype=torch.float32)

        if self.use_optical_flow:        
            all_five_parts_optical_flow = self.all_five_parts_optical_flow
            if self.data_type == 'casme3':
                n_img = f"{subject}_{filename}_{apex}.png"
            elif self.data_type == 'casme2':
                n_img = "sub"+str(subject).zfill(2)+f"_{filename} .png"
            # print(all_five_parts_optical_flow.keys())
            assert n_img in all_five_parts_optical_flow.keys(), f"Frame {n_img} not found in optical flow data."
            l_eye_lips = cv2.hconcat([all_five_parts_optical_flow[n_img][0], all_five_parts_optical_flow[n_img][1]]) # [14, 28, 3]
            r_eye_lips  =  cv2.hconcat([all_five_parts_optical_flow[n_img][3], all_five_parts_optical_flow[n_img][4]]) # [14, 28, 3]
            lr_eye_lips = cv2.vconcat([l_eye_lips, r_eye_lips]) # [28, 28, 3]

            lr_eye_lips = torch.tensor(lr_eye_lips, dtype=torch.float32) 
            lr_eye_lips = lr_eye_lips.permute(2, 0, 1) 
            return au_sequence, emotion, lr_eye_lips
        
        if self.use_img:
            if self.data_type == 'casme3':
                apex_frame_name =  os.path.join(frame_prefix, f"{apex}.jpg")
                onset_frame_name = os.path.join(frame_prefix, f"{onset}.jpg")
            elif self.data_type == 'casme2':
                apex_frame_name =  os.path.join('sub'+str(subject).zfill(2), filename, f"img{apex}.jpg")
                onset_frame_name = os.path.join('sub'+str(subject).zfill(2), filename, f"img{onset}.jpg")
            apex = cv2.cvtColor(cv2.imread(os.path.join(self.data_path, apex_frame_name)), cv2.COLOR_BGR2RGB)
            onset = cv2.cvtColor(cv2.imread(os.path.join(self.data_path, onset_frame_name)), cv2.COLOR_BGR2RGB)
            apex = cv2.resize(apex, (224, 224))
            onset = cv2.resize(onset, (224, 224))
            # cv2.imwrite('diff.jpg', np.abs(apex-onset))
            
            if self.first_transform is not None:
                
                apex = self.first_transform(apex)
                onset = self.first_transform(onset)
                all = torch.cat((onset, apex), 0)
                if self.img_transform is not None:
                    all = self.img_transform(all)
                onset = all[:3, :, :]
                apex  = all[3:, :, :]

            return au_sequence, emotion, onset, apex

        return au_sequence, emotion

    def augment_au(self, au_sequence):
        """
        Apply data augmentation to the AU sequence.
        """
        # Randomly choose an augmentation type
        augmentation_type = random.choice(['noise', 'scale', 'shift', 'dropout', 'interpolate'])
        if augmentation_type == 'noise':
            au_sequence = add_noise(au_sequence)
        elif augmentation_type == 'scale':
            au_sequence = scale_au(au_sequence)
        elif augmentation_type == 'shift':
            au_sequence = shift_au(au_sequence)
        elif augmentation_type == 'dropout':
            au_sequence = dropout_au(au_sequence)
        elif augmentation_type == 'interpolate':
            au_sequence = interpolate_au(au_sequence)


        # # Apply all augmentations
        # au_sequence = add_noise(au_sequence)
        # au_sequence = scale_au(au_sequence)
        # au_sequence = shift_au(au_sequence)
        # au_sequence = dropout_au(au_sequence)
        # au_sequence = interpolate_au(au_sequence)
        return au_sequence






# Example usage
if __name__ == "__main__":
    dataset = CASMEDataset(mode='test', test_subject='spNO.3')
    print(f"Dataset size: {len(dataset)}")
    au_values, emotion = dataset[1]
    print(f"AU values shape: {au_values.shape}")
    print(f"Emotion: {emotion}")

