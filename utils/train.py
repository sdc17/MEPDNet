import os
import json
import torch
import datetime
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from model import SegNet
from utils import get_range_limited_float_type
from utils import timewrapper, setup_logger, SingleDataset, SeqDataset, ToLabel, LongSeqDataset
from utils import CrossEntropyLoss2d
from torch.optim import SGD,Adam
from torch.utils.data import DataLoader,random_split


def eval(cfg, net, loader, device):
    target_type = torch.float32 if cfg['class_num'] == 1 else torch.long
    n_val = len(loader)
    tot = 0

    with tqdm(total=n_val,desc='Validation ',unit='batch',leave=False) as pbar:
        for iter, item in enumerate(loader):
            if cfg['seq']:
                imgu1s, imgs, imgd1s, targets = item
                imgu1s, imgs, imgd1s = imgu1s.to(device), imgs.to(device), imgd1s.to(device)
            elif cfg['longseq']:
                imgu2s, imgu1s, imgs, imgd1s, imgd2s, targets = item
                imgu2s, imgu1s, imgs, imgd1s, imgd2s = imgu2s.to(device), imgu1s.to(device), imgs.to(device), imgd1s.to(device), imgd2s.to(device)
            else:
                imgs, targets = item
                imgs = imgs.to(device)
            targets = targets.to(device)

            with torch.no_grad():
                if cfg['seq']:
                    predict = net(imgu1s, imgs, imgd1s)
                elif cfg['longseq']:
                    predict = net(imgu2s, imgu1s, imgs, imgd1s, imgd2s)
                else:
                    predict = net(imgs)
                
            if cfg['class_num'] > 1:
                weight = torch.tensor(cfg['weight']).to(device)
                tot += F.cross_entropy(predict,targets,weight).item()
            else:
                mask = torch.sigmoid(predict)
                mask = (mask>0.5).float()
                tot += dice_coeff(mask,targets).item()

            pbar.update()
    
    return tot/n_val


def train(cfg, net):
    lr = cfg['lr']
    mode = cfg['mode']
    device = cfg['device']
    weight = cfg['weight']
    train_img_dir = cfg['train_img_dir']
    train_mask_dir = cfg['train_mask_dir']
    val_img_dir = cfg['val_img_dir']
    val_mask_dir = cfg['val_mask_dir']
    batch_size = cfg['batch_size']
    num_workers = cfg['num_workers']
    input_transform = cfg['input_transform']
    target_transform = cfg['target_transform']
    model_name = net.__class__.__name__
    if len(cfg['gpu_ids']) > 1:
        model_name = net.module.__class__.__name__

    current_time = datetime.datetime.now()
    logger_file = os.path.join('log',mode,'{} {} lr {} bs {} ep {}.log'.
                    format(model_name,current_time.strftime('%Y%m%d %H:%M:%S'),
                            cfg['lr'],cfg['batch_size'],cfg['epochs']))
    logger = setup_logger(f'{model_name} {mode}',logger_file)

    if cfg['seq']:
        train_dataset = SeqDataset(train_img_dir, train_mask_dir, input_transform, target_transform, logger)
        val_dataset = SeqDataset(val_img_dir, val_mask_dir, input_transform, target_transform, logger)
    elif cfg['longseq']:
        train_dataset = LongSeqDataset(train_img_dir, train_mask_dir, input_transform, target_transform, logger)
        val_dataset = LongSeqDataset(val_img_dir, val_mask_dir, input_transform, target_transform, logger)
    else:
        train_dataset = SingleDataset(train_img_dir, train_mask_dir, input_transform, target_transform, logger)
        val_dataset = SingleDataset(val_img_dir, val_mask_dir, input_transform, target_transform, logger)

    n_val = len(val_dataset)
    n_train = len(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)

    display_weight = weight
    weight = torch.tensor(weight)
    if device == 'cuda':
        weight = weight.cuda()

    criterion = CrossEntropyLoss2d(weight=weight)

    optimizer = Adam(net.parameters(),lr=cfg['lr'],betas=(0.9,0.999))

    logger.info(f'''Starting training:
        Model:           {model_name}
        Epochs:          {cfg['epochs']}
        Batch size:      {cfg['batch_size']}
        Learning rate:   {cfg['lr']}
        Training size:   {n_train}
        Weight:          {display_weight}
        Validation size: {n_val}
        Device:          {device}
    ''')

    iter_num = 0
    train_batch_num = len(train_dataset)//batch_size
    for epoch in range(1,cfg['epochs']+1):
        net.train()
        for m in net.modules():
            if isinstance(net, nn.BatchNorm2d):
                m.track_running_stats=True
        epoch_loss = []
        logger.info('epoch[{}/{}]'.format(epoch,cfg['epochs']))
        with tqdm(total=n_train, desc='Epoch {}/{}'.format(epoch,cfg['epochs']),unit='imgs') as pbar:
            for iter, item in enumerate(train_loader):
                if cfg['seq']:
                    imgu1s, imgs, imgd1s, targets = item
                    imgu1s, imgs, imgd1s = imgu1s.to(device), imgs.to(device), imgd1s.to(device)
                    predict = net(imgu1s, imgs, imgd1s)
                elif cfg['longseq']:
                    imgu2s, imgu1s, imgs, imgd1s, imgd2s, targets = item
                    imgu2s, imgu1s, imgs, imgd1s, imgd2s = imgu2s.to(device), imgu1s.to(device), imgs.to(device), imgd1s.to(device), imgd2s.to(device)
                    predict = net(imgu2s, imgu1s, imgs, imgd1s, imgd2s)
                else:
                    imgs, targets = item
                    imgs = imgs.to(device)
                    predict = net(imgs)
                targets = targets.to(device)

                loss = criterion(predict,targets)

                loss_item = loss.item()
                epoch_loss.append(loss_item)
                pbar.set_postfix(**{'loss (batch)':loss_item})
                
                optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_value_(net.parameters(), 0.1) 
                optimizer.step()

                pbar.update(imgs.shape[0])
                iter_num += 1
                if iter_num % (train_batch_num//1) == 0:
                    net.eval()
                    val_score = eval(cfg,net,val_loader,device)

                    if cfg['class_num']>1:
                        logger.info('Validation cross entropy: {:.6f}'.format(val_score))
                    else:
                        logger.info('Validation Dice Coeff: {:.6f}'.format(val_score))

        if not os.path.exists(cfg['checkpoint_dir']):
            os.makedirs(cfg['checkpoint_dir'])
            logger.info('Created checkpoint directory:{}'.format(cfg['checkpoint_dir']))
        torch.save(net.state_dict(),os.path.join(cfg['checkpoint_dir'],"{}_{}.pth".format(cfg['model'],epoch)))
        logger.info(f'Checkpoint {epoch} saved!')
        print(' ')