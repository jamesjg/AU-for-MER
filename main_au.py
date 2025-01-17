from os import path
import os
import numpy as np
import cv2
import time

import pandas
from sklearn.metrics import confusion_matrix
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import argparse
from tqdm import tqdm
from distutils.util import strtobool
import torch
from Model import AUTransformer, HTNet_AU
import numpy as np
from facenet_pytorch import MTCNN
from Dataset import CASMEDataset

from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms




def reset_weights(m):  # Reset the weights for network to avoid weight leakage
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            #             print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()

def confusionMatrix(gt, pred):
    TN, FP, FN, TP = confusion_matrix(gt, pred).ravel()
    f1_score = (2 * TP) / (2 * TP + FP + FN)
    num_samples = len([x for x in gt if x == 1])
    average_recall = TP / num_samples
    return f1_score, average_recall


def recognition_evaluation(final_gt, final_pred, num_classes=7):
    if num_classes == 7:
        label_dict = {'others': 0, 'anger': 1, 'disgust': 2, 'fear': 3, 'happy': 4, 'sad': 5, 'surprise': 6}
    elif num_classes == 3:
        label_dict = {'negative': 0, 'positive': 1, 'surprise': 2}

    # Display recognition result
    f1_list = []
    ar_list = []
    try:
        for emotion, emotion_index in label_dict.items():
            gt_recog = [1 if x == emotion_index else 0 for x in final_gt]
            pred_recog = [1 if x == emotion_index else 0 for x in final_pred]
            try:
                f1_recog, ar_recog = confusionMatrix(gt_recog, pred_recog)
                f1_list.append(f1_recog)
                ar_list.append(ar_recog)
            except Exception as e:
                pass
        UF1 = np.mean(f1_list)
        UAR = np.mean(ar_list)
        return UF1, UAR
    except:
        return '', ''

def collate_fn(batch, pad_value=0.0):
    au_sequences, emotion, img = zip(*batch)
    max_length = max(seq.shape[0] for seq in au_sequences)
    padded_sequences = []
    for seq in au_sequences:
        pad_len = max_length - seq.shape[0]
        if pad_len > 0:
            padding = torch.full((pad_len, seq.shape[1]), pad_value, dtype=seq.dtype)
            seq = torch.cat((seq, padding), dim=0)
        padded_sequences.append(seq)
    padded_sequences = torch.stack(padded_sequences)
    emotion = torch.tensor(emotion, dtype=torch.long)
    img = torch.stack(img)

    return padded_sequences, emotion, img




def main(args):
    learning_rate = args.lr
    batch_size = args.batch_size
    epochs = args.epoch
    all_accuracy_dict = {}
    is_cuda = torch.cuda.is_available()

    #seed
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)  # gpu
    np.random.seed(args.random_seed)  # numpy

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True


    if is_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')


    total_gt = []
    total_pred = []
    best_total_pred = []

    t = time.time()

    # For CASME
    main_path = args.optical_flow_path
    subName = os.listdir(main_path)
    subName = [x for x in subName if x.endswith('.png') or x.endswith('.jpg')]
    subjects = [x.split('_')[0] for x in subName]
    subjects = sorted(list(set(subjects)))
    print(subjects)

    first_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_transform = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(5),
        transforms.RandomCrop(224, padding=4),
    ])

    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    for subject in tqdm(subjects):
        train_dataset = CASMEDataset(mode='train', test_subject=subject, num_classes=args.num_classes, 
                                     json_path=args.json_path,
                                     data_path=args.data_path,
                                     xlsx_path=args.xlsx_path,
                                     data_type=args.data_type,
                                     transform_au=args.transform_au,
                                     first_transform=first_transform,
                                     img_transform=train_transform)
        test_dataset = CASMEDataset(mode='test', test_subject=subject, num_classes=args.num_classes, 
                                    json_path=args.json_path,
                                    data_path=args.data_path,
                                    xlsx_path=args.xlsx_path,
                                    data_type=args.data_type,
                                    first_transform=test_transform)

        if len(test_dataset) == 0:
            print(f'Subject: {subject} | No test data')
            continue

        sample_weights = train_dataset.get_sample_weights()
        sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, sampler=sampler)
        
        # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        test_dataloader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=8)
        
        print(f'Subject: {subject} | Train Size: {len(train_dataset)} | Test Size: {len(test_dataset)}')

        model = HTNet_AU(            
            image_size=28,
            patch_size=7,
            dim=256,  # 256,--96, 56-, 192
            heads=3,  # 3 ---- , 6-
            num_hierarchies=3,  # 3----number of hierarchies
            block_repeats=(2, 2, 8),#(2, 2, 8),------
            # the number of transformer blocks at each heirarchy, starting from the bottom(2,2,20) -
            num_classes=args.num_classes,
            dropout=args.dropout,)
        
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate, weight_decay=args.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        #weight = train_dataset.get_class_weights()
        # loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        # print(train_dataset.get_class_weights())
        loss_fn = nn.CrossEntropyLoss()


        # store best results
        best_accuracy_for_each_subject = -1
        best_each_subject_pred = []

        for epoch in (range(epochs)):
            # Training
            model.train()
            train_loss = 0.0
            num_train_correct = 0
            num_train_examples = 0

            for au_sequence, emotion, img in train_dataloader:
                au_sequence = au_sequence.to(device)
                emotion = emotion.to(device)
                # print(sum(emotion==0), sum(emotion==1), sum(emotion==2))
                img = img.to(device)
                # apex = apex.to(device)
                # onset = onset.to(device)
                optimizer.zero_grad()
                pred_emotion = model(au_sequence, img)
                loss = loss_fn(pred_emotion, emotion)
                loss.backward()
                optimizer.step()
                train_loss += loss.data.item() * au_sequence.size(0)
                num_train_correct += (torch.max(pred_emotion, 1)[1] == emotion).sum().item()
                num_train_examples += emotion.shape[0]
            scheduler.step()

            train_acc = num_train_correct / num_train_examples
            train_loss = train_loss / len(train_dataloader.dataset)


            # Testing
            model.eval()
            test_loss = 0.0
            num_test_correct = 0
            num_test_examples = 0
            test_pred = []
            test_gt = []
            with torch.no_grad():
                for au_sequence, emotion, img in test_dataloader:
                    au_sequence = au_sequence.to(device)
                    emotion = emotion.to(device)
                    img = img.to(device)
                    # onset = onset.to(device)
                    # apex = apex.to(device)
                    pred_emotion = model(au_sequence, img)
                    loss = loss_fn(pred_emotion, emotion)
                    test_loss += loss.data.item() * au_sequence.size(0)
                    num_test_correct += (torch.max(pred_emotion, 1)[1] == emotion).sum().item()
                    num_test_examples += emotion.shape[0]
                    # test_pred.extend(torch.max(pred_emotion, 1)[1].tolist())
                    # test_gt.extend(emotion.tolist())

            test_acc = num_test_correct / num_test_examples
            test_loss = test_loss / len(test_dataloader.dataset)

            print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f} | Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f} | LR: {optimizer.param_groups[0]['lr']}")

            temp_each_subject_pred = []
            if test_acc >= best_accuracy_for_each_subject:
                best_accuracy_for_each_subject = test_acc
                temp_each_subject_pred.extend(torch.max(pred_emotion, 1)[1].tolist())
                best_each_subject_pred = temp_each_subject_pred

        assert len(test_dataset) == len(best_each_subject_pred) # test_batch_size must bigger than len(test_dataset)
        # For UF1 and UAR computation
        print('Best Predicted    :', best_each_subject_pred)
        accuracydict = {}
        accuracydict['pred'] = best_each_subject_pred
        accuracydict['truth'] = emotion.tolist()
        all_accuracy_dict[subject] = accuracydict

        print('Ground Truth :', emotion.tolist())
        print('Predicted    :', torch.max(pred_emotion, 1)[1].tolist())
        print("Evaluation until this subject: ")
        total_pred.extend(torch.max(pred_emotion, 1)[1].tolist())
        total_gt.extend(emotion.tolist())
        best_total_pred.extend(best_each_subject_pred)
        UF1, UAR = recognition_evaluation(total_gt, total_pred, num_classes=args.num_classes)
        best_UF1, best_UAR = recognition_evaluation(total_gt, best_total_pred, num_classes=args.num_classes)
        print('UF1:', round(UF1, 4), '| UAR:', round(UAR, 4))
        print('Best UF1:', round(best_UF1, 4), '| Best UAR:', round(best_UAR, 4))

    print('Final Evaluation: ')
    UF1, UAR = recognition_evaluation(total_gt, total_pred, num_classes=args.num_classes)
    best_UF1, best_UAR = recognition_evaluation(total_gt, best_total_pred, num_classes=args.num_classes)
    print(np.shape(total_gt))
    print('Total Time Taken:', time.time() - t)
    print(all_accuracy_dict)
    print('UF1:', round(UF1, 4), '| UAR:', round(UAR, 4))
    print('Best UF1:', round(best_UF1, 4), '| Best UAR:', round(best_UAR, 4))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str, default='datasets/CASME3/ME_inference_au.json')
    parser.add_argument('--xlsx_path', type=str, default='datasets/CASME3/annotation/cas(me)3_part_A_ME_label_JpgIndex_v2_final.xlsx')
    parser.add_argument('--data_type', type=str, default='casme3')
    parser.add_argument('--optical_flow_path', type=str, default='datasets/casme3_crop_tvl1_whole_norm_u_v_os')
    parser.add_argument('--data_path', type=str, default='datasets/CASME3/Part_A_ME_clip_scrfd_cropped')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--transform_au', action='store_true')


    args = parser.parse_args()
    main(args)
