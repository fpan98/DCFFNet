import os
from collections import OrderedDict
import random
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim

import torch.utils.data as Data
from torch.optim import lr_scheduler
from tqdm import tqdm
from utils.utils import iou_score, score
from utils.loss import dice_bce_loss

import utils.dataset as dates
from torch.utils.tensorboard import SummaryWriter
from arch.m2fnet import IFN_NestedUnet

model_name = "M2F-Net"


def train(train_loader, model, criterion, optimizer):
    precision = 0.0
    recall = 0.0
    F1 = 0.0
    OA = 0.0
    num = 0
    total_loss = 0.0
    iou = 0.0

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    model.train()
    pbar = tqdm(total=len(train_loader))
    for input1, input2, target in train_loader:
        outputs = model(input1.cuda(), input2.cuda())
        target = target.cuda()
        loss = 0
        for output in outputs:
            output = torch.clamp(output, 0.0001, 0.9999)
            loss += criterion(output, target)
        loss = loss / len(outputs)
        output = outputs[-1]
        for i in range(target.shape[0]):
            iou += iou_score(output[i], target[i])
            num += 1
        res = score(output, target)
        TP += res[0]
        TN += res[1]
        FP += res[2]
        FN += res[3]

        if TP != 0:
            precision = TP * 1.0 / (TP + FP)
            recall = TP * 1.0 / (TP + FN)
            F1 = 2 * precision * recall / (precision + recall)
            OA = (TP + TN) * 1.0 / (TP + TN + FP + FN)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        postfix = OrderedDict([
            ('loss', total_loss / num),
            ('iou', iou / num),
            ('P', precision),
            ('R', recall),
            ('F1', F1),
            ('OA', OA),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', total_loss / num),
                        ('iou', iou / num),
                        ('P', precision),
                        ('R', recall),
                        ('F1', F1),
                        ('OA', OA)])


def validate(val_loader, model, criterion):
    precision = 0.0
    recall = 0.0
    F1 = 0.0
    OA = 0.0
    num = 0
    iou = 0.0
    total_loss = 0.0

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    model.eval()
    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input1, input2, target in val_loader:
            outputs = model(input1.cuda(), input2.cuda())
            target = target.cuda()
            loss = 0
            for output in outputs:
                output = torch.clamp(output, 0.0001, 0.9999)
                loss += criterion(output, target)
            loss = loss / len(outputs)
            output = outputs[-1]
            for i in range(target.shape[0]):
                iou += iou_score(output[i], target[i])  # output5计算IOU
                num += 1

            res = score(output, target)
            TP += res[0]
            TN += res[1]
            FP += res[2]
            FN += res[3]

            if TP != 0:
                precision = TP * 1.0 / (TP + FP)
                recall = TP * 1.0 / (TP + FN)
                F1 = 2 * precision * recall / (precision + recall)
                OA = (TP + TN) * 1.0 / (TP + TN + FP + FN)

            total_loss += loss.item()
            postfix = OrderedDict([
                ('loss', total_loss / num),
                ('iou', iou / num),
                ('P', precision),
                ('R', recall),
                ('F1', F1),
                ('OA', OA)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

        return OrderedDict([('loss', total_loss / num),
                            ('iou', iou / num),
                            ('P', precision),
                            ('R', recall),
                            ('F1', F1),
                            ('OA', OA)])


def main():
    epochs = 30
    learning_rate = 0.00005
    batch_size = 4
    criterion = dice_bce_loss()
    model = IFN_NestedUnet()
    if torch.cuda.is_available():
        cudnn.benchmark = True
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.90)

    base_path = os.path.join(r'F:\ChangeDetectionDataset\Real\subset')
    train_txt_path = os.path.join(r'F:\ChangeDetectionDataset\Real\subset', 'train2.txt')
    val_txt_path = os.path.join(r'F:\ChangeDetectionDataset\Real\subset', 'val2.txt')

    train_data = dates.Dataset(base_path, train_txt_path)
    train_loader = Data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_data = dates.Dataset(base_path, val_txt_path)
    val_loader = Data.DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('precision', []),
        ('recall', []),
        ('F1', []),
        ('OA', []),
        ('val_loss', []),
        ('val_iou', []),
        ('val_precision', []),
        ('val_recall', []),
        ('val_F1', []),
        ('val_OA', []),
    ])
    best_f1 = 0.8

    for epoch in range(epochs):
        print('Epoch [%d/%d]' % (epoch + 1, epochs))
        # train for one epoch
        train_log = train(train_loader, model, criterion, optimizer)
        # evaluate on validation set
        val_log = validate(val_loader, model, criterion)
        # 学习率调整
        scheduler.step()
        print('train:loss %.4f - iou %.4f - precision %.4f - recall %.4f - F1 %.4f - OA %.4f\n '
              'val: loss %.4f - iou %.4f- precision %.4f - recall %.4f - F1 %.4f - OA %.4f '
            % (train_log['loss'], train_log['iou'], train_log['P'], train_log['R'], train_log['F1'], train_log['OA'],
               val_log['loss'], val_log['iou'], val_log['P'], val_log['R'], val_log['F1'], val_log['OA']))

        # tensorboard可视化
        log_dir = "models/"+model_name+"/tensorboard"
        writer = SummaryWriter(log_dir=log_dir)
        writer.add_scalars("loss", {"train_loss": train_log['loss'],
                                    "val_loss": val_log['loss']
                                    }, epoch + 1)
        writer.add_scalars("iou", {"train_iou": train_log['iou'],
                                   "val_iou": val_log['iou']}, epoch + 1)
        writer.add_scalars("Precision", {"train_P": train_log['P'],
                                   "val_P": val_log['P']}, epoch + 1)
        writer.add_scalars("Recall", {"train_R": train_log['R'],
                                   "val_R": val_log['R']}, epoch + 1)
        writer.add_scalars("F1", {"train_F1": train_log['F1'],
                                   "val_F1": val_log['F1']}, epoch + 1)
        writer.add_scalars("OA", {"train_OA": train_log['OA'],
                                   "val_OA": val_log['OA']}, epoch + 1)

        # 保存日志信息
        log['epoch'].append(epoch)
        log['lr'].append(optimizer.param_groups[0]["lr"])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['precision'].append(train_log['P'])
        log['recall'].append(train_log['R'])
        log['F1'].append(train_log['F1'])
        log['OA'].append(train_log['OA'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_precision'].append(val_log['P'])
        log['val_recall'].append(val_log['R'])
        log['val_F1'].append(val_log['F1'])
        log['val_OA'].append(val_log['OA'])

        pd.DataFrame(log).to_csv('models/%s/log.csv' % model_name, index=False)

        if val_log['F1'] > best_f1:
            torch.save(model.state_dict(), 'models/%s/model.pth' % model_name)
            best_f1 = val_log['F1']
            print("=> saved best model")

        torch.cuda.empty_cache()  # 清除没有用的临时变量


# 随机数种子确定时，模型的训练结果将保持一致
def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    seed_torch(777)
    main()