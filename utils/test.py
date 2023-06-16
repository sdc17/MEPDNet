import os
import cv2
import glob
import torch
import shutil
import datetime
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from tqdm import tqdm
from model import SegNet
from os.path import split,splitext
from utils import setup_logger, is_image_file, SeqTestDataset, SingleTestDataset, LongSeqTestDataset
from torch.utils.data import DataLoader

from timeit import default_timer as timer


def calDSC(predict,target):
    eps = 0.0001
    predict = predict.astype(np.float32)
    target = target.astype(np.float32)
    intersection = np.dot(predict.reshape(-1),target.reshape(-1))
    union = predict.sum() + target.sum() + eps
    dsc = (2 * intersection + eps) / union
    return dsc


def calRecall(predict, target):
    eps = 0.0001
    predict = predict.astype(np.int8).reshape(-1)
    target = target.astype(np.int8).reshape(-1)
    TP = (predict == 1) & (target == 1)
    FN = (predict == 0) & (target == 1)
    recall = TP.sum() / (TP.sum() + FN.sum() + eps)
    return recall


def calPrecision(predict, target):
    eps = 0.0001
    predict = predict.astype(np.int8).reshape(-1)
    target = target.astype(np.int8).reshape(-1)
    TP = (predict == 1) & (target == 1)
    FP = (predict == 1) & (target == 0)
    precision = TP.sum() / (TP.sum() + FP.sum() + eps)
    return precision


def calDistance(seg_A, seg_B, dx=1.0, k=1):
    # Extract the label k from the segmentation maps to generate binary maps
    seg_A = (seg_A == k)
    seg_B = (seg_B == k)

    table_md = []
    table_hd = []

    X, Y, Z = seg_A.shape

    for z in range(Z):
        # Binary mask at this slice
        slice_A = seg_A[:, :, z].astype(np.uint8)
        slice_B = seg_B[:, :, z].astype(np.uint8)

        # The distance is defined only when both contours exist on this slice
        if np.sum(slice_A) > 0 and np.sum(slice_B) > 0:
            # Find contours and retrieve all the points
            contours, _ = cv2.findContours(cv2.inRange(slice_A, 1, 1),
                                              cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_NONE)
            pts_A = contours[0]
            for i in range(1, len(contours)):
                pts_A = np.vstack((pts_A, contours[i]))

            contours, _ = cv2.findContours(cv2.inRange(slice_B, 1, 1),
                                              cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_NONE)
            pts_B = contours[0]
            for i in range(1, len(contours)):
                pts_B = np.vstack((pts_B, contours[i]))

            # Distance matrix between point sets
            M = np.zeros((len(pts_A), len(pts_B)))
            for i in range(len(pts_A)):
                for j in range(len(pts_B)):
                    M[i, j] = np.linalg.norm(pts_A[i, 0] - pts_B[j, 0])

            # Mean distance and hausdorff distance
            md = 0.5 * (np.mean(np.min(M, axis=0)) + np.mean(np.min(M, axis=1))) * dx
            hd = np.max([np.max(np.min(M, axis=0)), np.max(np.min(M, axis=1))]) * dx
            table_md += [md]
            table_hd += [hd]

    # Return the mean distance and Hausdorff distance across 2D slices
    mean_md = np.mean(table_md) if table_md else None
    mean_hd = np.mean(table_hd) if table_hd else None
    return mean_md, mean_hd


def test(cfg,net,model_path):
    mode = cfg['mode']
    device = cfg['device']
    class_num = cfg['class_num']
    batch_size = cfg['batch_size']
    num_workers = cfg['num_workers']
    test_img_dir = cfg['test_img_dir']
    test_mask_dir = cfg['test_mask_dir']

    model_name = net.__class__.__name__
    if len(cfg['gpu_ids']) > 1:
        model_name = net.module.__class__.__name__

    # performance_dir = os.path.join('test_performance', model_name)
    # if os.path.exists(performance_dir):
    #     shutil.rmtree(performance_dir)
    # os.makedirs(performance_dir,exist_ok=True)

    performance_dir = os.path.join('test_performance_add', model_name)
    if os.path.exists(performance_dir):
        shutil.rmtree(performance_dir)
    os.makedirs(performance_dir,exist_ok=True)

    performance_dir = os.path.join('test_performance_add', 'GT')
    if os.path.exists(performance_dir):
        shutil.rmtree(performance_dir)
    os.makedirs(performance_dir,exist_ok=True)

    current_time = datetime.datetime.now()
    logger_file = os.path.join('log',mode,'{} {}.log'.
                    format(model_name,current_time.strftime('%Y%m%d %H:%M:%S')))
    logger = setup_logger(f'{model_name} {mode}',logger_file)
    
    if cfg['seq']:
        dataset = SeqTestDataset(test_img_dir, test_mask_dir, cfg['input_transform'], cfg['target_transform'], logger)
    elif cfg['longseq']:
        dataset = LongSeqTestDataset(test_img_dir, test_mask_dir, cfg['input_transform'], cfg['target_transform'], logger)
    else:
        dataset = SingleTestDataset(test_img_dir, test_mask_dir, cfg['input_transform'], cfg['target_transform'], logger)
    loader = DataLoader(dataset, batch_size=batch_size,shuffle=False,num_workers=num_workers)

    net.load_state_dict(torch.load(model_path,map_location=device))

    dsc = []
    recall = []
    precision = []
    distance = []

    net.eval()
    for iter, item in enumerate(loader):
        if cfg['seq']:
            imgu1s, imgs, imgd1s, targets, filenames = item
            imgu1s, imgs, imgd1s = imgu1s.to(device), imgs.to(device), imgd1s.to(device)
        elif cfg['longseq']:
            imgu2s, imgu1s, imgs, imgd1s, imgd2s, targets, filenames = item
            imgu2s, imgu1s, imgs, imgd1s, imgd2s = imgu2s.to(device), imgu1s.to(device), imgs.to(device), imgd1s.to(device), imgd2s.to(device)
        else:
            imgs, targets, filenames = item
            imgs = imgs.to(device)
    
        with torch.no_grad():
            if cfg['seq']:
                predict = net(imgu1s, imgs, imgd1s)
            elif cfg['longseq']:
                predict = net(imgu2s, imgu1s, imgs, imgd1s, imgd2s)
            else:
                predict = net(imgs)

        if class_num > 1:
            probs = F.softmax(predict,dim=1)
        else:
            probs = torch.sigmoid(predict)
            
        masks = torch.argmax(probs,dim=1).cpu().numpy()
            
        for i, file_name in enumerate(filenames):
            img = cv2.imread(file_name)
            mask = masks[i].astype(np.uint8)
            ground_truth = targets[i].numpy().astype(np.uint8)
            # contours, hierarchy = cv2.findContours(ground_truth, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            # for contour in contours:
            #     for point in contour:
            #         x, y = point[0]
            #         img[y, x, :] = [0, 0, 255]

            # contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            # for contour in contours:
            #     for point in contour:
            #         x, y = point[0]
            #         img[y, x, :] = [255, 255, 0]
                
            dsc.append(calDSC(mask, ground_truth))
            recall.append(calRecall(mask, ground_truth))
            precision.append(calPrecision(mask, ground_truth))
            # distance.append(calDistance(np.expand_dims(mask, -1), np.expand_dims(ground_truth, -1))[0])

            # dst = os.path.join('test_performance', model_name, file_name.split('/')[-1]).replace('.jpg', '.png')
            # cv2.imwrite(dst, img)

            dst = os.path.join('test_performance_add', model_name, file_name.split('/')[-1]).replace('.jpg', '.png')
            colors = np.full(img.shape, [159, 255, 84], dtype=np.uint8) \
                * np.array([mask for i in range(3)]).transpose(1,2,0) \
                # + np.full(img.shape, [0, 255, 0], dtype=np.uint8) \
                # * (1 - np.array([ground_truth for i in range(3)])).transpose(1,2,0)
            cv2.imwrite(dst, cv2.addWeighted(img, 0.84, colors, 0.16, 0))

            dst = os.path.join('test_performance_add', 'GT', file_name.split('/')[-1]).replace('.jpg', '.png')
            colors = np.full(img.shape, [159, 255, 84], dtype=np.uint8) \
                * np.array([ground_truth for i in range(3)]).transpose(1,2,0) \
                # + np.full(img.shape, [0, 255, 0], dtype=np.uint8) \
                # * (1 - np.array([ground_truth for i in range(3)])).transpose(1,2,0)
            cv2.imwrite(dst, cv2.addWeighted(img, 0.84, colors, 0.16, 0))
        
        # weight = torch.tensor(cfg['weight']).to(device)
        # print(F.cross_entropy(probs, targets.to(device), weight).item())
        # np.savetxt('temp.txt', masks[0], fmt='%f',delimiter=',')
        # np.savetxt('temp.txt', probs[0][0].cpu().numpy(), fmt='%f',delimiter=',')
        
    dsc_10 = [item for i, item in enumerate(dsc) if 0 <= (i % 30) < 10]
    dsc_20 = [item for i, item in enumerate(dsc) if 10 <= (i % 30) < 20]
    dsc_30 = [item for i, item in enumerate(dsc) if 20 <= (i % 30)]
    # print('{}|DSC: {:.2f} 0~10: {:.2f} 10~20: {:.2f} 20~30: {:.2f} Recall: {:.2f} Precision: {:.2f} Distance: {:.2f}'
    #     .format(model_name, np.mean(dsc)*100, np.mean(dsc_10)*100, np.mean(dsc_20)*100, np.mean(dsc_30)*100, np.mean(recall)*100, np.mean(precision)*100, np.mean(distance)))
    print('{}|DSC: {:.2f} 0~10: {:.2f} 10~20: {:.2f} 20~30: {:.2f} Recall: {:.2f} Precision: {:.2f}'
        .format(model_name, np.mean(dsc)*100, np.mean(dsc_10)*100, np.mean(dsc_20)*100, np.mean(dsc_30)*100, np.mean(recall)*100, np.mean(precision)*100))
    # print('DSC of {}: {}'.format(model_name, np.mean(dsc)))
        