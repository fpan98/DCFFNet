import cv2


def score_LEVIR(output, target, threshold=0.5):
    TP = ((output >= threshold) & (target >= threshold)).sum()
    TN = ((output < threshold) & (target < threshold)).sum()
    FP = ((output >= threshold) & (target < threshold)).sum()
    FN = ((output < threshold) & (target >= threshold)).sum()
    return [int(TP), int(TN), int(FP), int(FN)]


def score_CDD(output, target, threshold=0.5):
    TP = ((output >= threshold) & (target >= threshold)).sum()
    TN = ((output < threshold) & (target < threshold)).sum()
    FP = ((output >= threshold) & (target < threshold)).sum()
    FN = ((output < threshold) & (target >= threshold)).sum()
    precision = TP * 1.0 / (max(TP + FP, 1e-4))
    recall = TP * 1.0 / (max(TP + FN, 1e-4))
    F1 = 2 * precision * recall / (max(precision + recall, 1e-4))
    OA = (TP + TN) * 1.0 / (max(TP + TN + FP + FN, 1e-4))
    return [precision, recall, F1, OA]


def iou_score(output, target):
    smooth = 1e-5
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    return float((intersection + smooth) / (union + smooth))


def show(output, id=None):
    b = output.shape[0]  # 对应batch的个数
    for i in range(b):
        change_map = output[i, :, :, :]  # [channel, width, height]
        change_map = change_map[0, :, :]  # [width, height]
        change_map = change_map >= 0.5
        change_map = (change_map * 255).detach().cpu().numpy().astype('uint8')
        # cv2.imwrite("./outputs/"+str(id)+".jpg", change_map)
        cv2.namedWindow("change_map", cv2.WINDOW_NORMAL)
        cv2.imshow("change_map", change_map)
        cv2.waitKey(0)