import torch
import cv2
import utils.dataset as dates
import os
import torch.utils.data as Data
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from arch.m2fnet import IFN_NestedUnet
from utils.utils import show, score


def predict():
    batch_size = 1
    model = IFN_NestedUnet()
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model = model.cuda()

    model = model.eval()
    model_path = r'models/M2F-Net/model.pth'
    model_dict = torch.load(model_path, map_location='cuda')
    model.load_state_dict(model_dict)
    print('Loading weights into state dict...')

    base_path = os.path.join(r'F:\ChangeDetectionDataset\Real\subset')
    test_txt_path = os.path.join(r'F:\ChangeDetectionDataset\Real\subset', 'test.txt')
    test_data = dates.Dataset(base_path, test_txt_path)
    test_loader = Data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    print(len(test_loader))
    num = 0.0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    with torch.no_grad():
        for input1, input2, target in tqdm(test_loader, total=len(test_loader)):  # target[batch, weight, height]
            output = model(input1, input2)
            output = output[-1]
            # 可视化
            # show(output, int(num+1))
            output = output.cpu().numpy()
            target = target.numpy()
            num = num + 1
            res = score(output, target)
            TP += res[0]
            TN += res[1]
            FP += res[2]
            FN += res[3]

    precision = TP * 1.0 / (TP + FP)
    recall = TP * 1.0 / (TP + FN)
    F1 = 2 * precision * recall / (precision + recall)
    OA = (TP + TN) * 1.0 / (TP + TN + FP + FN)

    print('precision: %.4f' % precision)
    print('recall: %.4f' % recall)
    print('F1: %.4f' % F1)
    print('OA: %.4f' % OA)


if __name__ == '__main__':
    predict()

