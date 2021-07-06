import os
import pickle
import numpy as np
import sys
from copy import deepcopy

"""We modify the test code from ACT. https://github.com/vkalogeiton/caffe/blob/act-detector/act-detector-scripts/ACT.py"""

""" boxes are represented as a numpy array with 4 columns corresponding to the coordinates (x1, y1, x2, y2)"""

def area2d(b):
    """Compute the areas for a set of 2D boxes"""

    return (b[:, 2] - b[:, 0] + 1) * (b[:, 3] - b[:, 1] + 1)

def overlap2d(b1, b2):
    """Compute the overlaps between a set of boxes b1 and one box b2"""

    xmin = np.maximum(b1[:, 0], b2[:, 0])
    ymin = np.maximum(b1[:, 1], b2[:, 1])
    xmax = np.minimum(b1[:, 2] + 1, b2[:, 2] + 1)
    ymax = np.minimum(b1[:, 3] + 1, b2[:, 3] + 1)

    width = np.maximum(0, xmax - xmin)
    height = np.maximum(0, ymax - ymin)

    return width * height

def iou2d(b1, b2):
    """Compute the IoU between a set of boxes b1 and 1 box b2"""

    if b1.ndim == 1:
        b1 = b1[None, :]
    if b2.ndim == 1:
        b2 = b2[None, :]

    assert b2.shape[0] == 1

    ov = overlap2d(b1, b2)

    return ov / (area2d(b1) + area2d(b2) - ov)

# TUBELETS
""" tubelets of length K are represented using numpy array with 4K columns """

# TUBES
""" tubes are represented as a numpy array with nframes rows and 5 columns (frame, x1, y1, x2, y2). frame number are 1-indexed, coordinates are 0-indexed """


def iou3d(b1, b2):
    """Compute the IoU between two tubes with same temporal extent"""

    assert b1.shape[0] == b2.shape[0]
    assert np.all(b1[:, 0] == b2[:, 0])

    ov = overlap2d(b1[:, 1:5], b2[:, 1:5])

    return np.mean(ov / (area2d(b1[:, 1:5]) + area2d(b2[:, 1:5]) - ov))


def iou3dt(b1, b2, spatialonly=False):
    """Compute the spatio-temporal IoU between two tubes"""

    tmin = max(b1[0, 0], b2[0, 0])
    tmax = min(b1[-1, 0], b2[-1, 0])

    if tmax < tmin:
        return 0.0

    temporal_inter = tmax - tmin + 1
    temporal_union = max(b1[-1, 0], b2[-1, 0]) - min(b1[0, 0], b2[0, 0]) + 1

    tube1 = b1[int(np.where(b1[:, 0] == tmin)[0]): int(np.where(b1[:, 0] == tmax)[0]) + 1, :]
    tube2 = b2[int(np.where(b2[:, 0] == tmin)[0]): int(np.where(b2[:, 0] == tmax)[0]) + 1, :]

    return iou3d(tube1, tube2) * (1. if spatialonly else temporal_inter / temporal_union)

# AP

def pr_to_ap(pr):
    """Compute AP given precision-recall
    pr is a Nx2 array with first row being precision and second row being recall
    """

    prdif = pr[1:, 1] - pr[:-1, 1]
    prsum = pr[1:, 0] + pr[:-1, 0]

    return np.sum(prdif * prsum * 0.5)

def frameAP(GT, submit_dir, th, print_info=True):

    vlist = GT['test_videos'][0]
    # load per-frame detections
    frame_detections_file = os.path.join(submit_dir, 'frame_detections.pkl')
    with open(frame_detections_file, 'rb') as fid:
        alldets = pickle.load(fid)

    results = {}
    # compute AP for each class
    for ilabel, label in enumerate(GT['labels']):
        # detections of this class
        if label in ['aerobic kick jump', 'aerobic off axis jump', 'aerobic butterfly jump', 'aerobic balance turn','basketball save','basketball jump ball']:
            print('do not evaluate{}'.format(label))
            continue
        detections = alldets[alldets[:, 2] == ilabel, :]

        # load ground-truth of this class
        gt = {}
        for iv, v in enumerate(vlist):
            tubes = GT['gttubes'][v]
            if ilabel not in tubes:
                continue

            for tube in tubes[ilabel]:
                for i in range(tube.shape[0]):
                    k = (iv, int(tube[i, 0]))
                    if k not in gt:
                        gt[k] = []
                    gt[k].append(tube[i, 1:5].tolist())

        for k in gt:
            gt[k] = np.array(gt[k])

        # pr will be an array containing precision-recall values
        pr = np.empty((detections.shape[0] + 1, 2), dtype=np.float32)  # precision,recall
        pr[0, 0] = 1.0
        pr[0, 1] = 0.0
        fn = sum([g.shape[0] for g in gt.values()])  # false negatives
        if fn==0:
            print('no such label',ilabel,label)
            continue
        fp = 0  # false positives
        tp = 0  # true positives

        for i, j in enumerate(np.argsort(-detections[:, 3])):
            k = (int(detections[j, 0]), int(detections[j, 1]))
            box = detections[j, 4:8]
            ispositive = False

            if k in gt:
                ious = iou2d(gt[k], box)
                amax = np.argmax(ious)

                if ious[amax] >= th:
                    ispositive = True
                    gt[k] = np.delete(gt[k], amax, 0)

                    if gt[k].size == 0:
                        del gt[k]

            if ispositive:
                tp += 1
                fn -= 1
            else:
                fp += 1

            pr[i + 1, 0] = float(tp) / float(tp + fp)
            pr[i + 1, 1] = float(tp) / float(tp + fn)

        results[label] = pr

    ap = 100 * np.array([pr_to_ap(results[label]) for label in results])
    class_result={}
    for label in results:
        class_result[label]=pr_to_ap(results[label])
    frameap_result = np.mean(ap)
    if print_info:
        print('frameAP_{}\n'.format( th))
        for l in class_result:
            print("{:20s} {:8.2f}".format(l,class_result[l]))
        print("{:20s} {:8.2f}".format("mAP", frameap_result))
    return frameap_result


def videoAP(GT, submit_dir, th, print_info=True):

    vlist = GT['test_videos'][0]
    video_detections_file = os.path.join(submit_dir, 'video_detections.pkl')
    with open(video_detections_file, 'rb') as fid:
        alldets = pickle.load(fid)
    # compute AP for each class
    res = {}
    for ilabel in range(len(GT['labels'])):
        if GT['labels'][ilabel] in ['aerobic kick jump', 'aerobic off axis jump', 'aerobic butterfly jump', 'aerobic balance turn','basketball save','basketball jump ball']:
            print('do not evaluate{}'.format(GT['labels'][ilabel]))
            continue
        detections = alldets[ilabel]
        # load ground-truth
        gt = {}
        for v in vlist:
            tubes = GT['gttubes'][v]

            if ilabel not in tubes:
                continue

            gt[v] = tubes[ilabel]

            if len(gt[v]) == 0:
                del gt[v]

        # precision,recall
        pr = np.empty((len(detections) + 1, 2), dtype=np.float32)
        pr[0, 0] = 1.0
        pr[0, 1] = 0.0

        fn = sum([len(g) for g in gt.values()])  # false negatives
        fp = 0  # false positives
        tp = 0  # true positives
        if fn==0:
            print('no such label', ilabel, GT['labels'][ilabel])
            continue
        for i, j in enumerate(np.argsort(-np.array([dd[1] for dd in detections]))):
            v, score, tube = detections[j]
            ispositive = False

            if v in gt:
                ious = [iou3dt(g, tube) for g in gt[v]]
                amax = np.argmax(ious)
                if ious[amax] >= th:
                    ispositive = True
                    del gt[v][amax]
                    if len(gt[v]) == 0:
                        del gt[v]

            if ispositive:
                tp += 1
                fn -= 1
            else:
                fp += 1

            pr[i + 1, 0] = float(tp) / float(tp + fp)
            pr[i + 1, 1] = float(tp) / float(tp + fn)

        res[GT['labels'][ilabel]] = pr

    ap = 100 * np.array([pr_to_ap(res[label]) for label in res])
    videoap_result = np.mean(ap)
    class_result={}
    for label in res:
        class_result[label]=pr_to_ap(res[label])
    if print_info:
        print('VideoAP_{}\n'.format(th))
        for l in class_result:
            print("{:20s} {:8.2f}".format(l,class_result[l]))
        print("{:20s} {:8.2f}".format("mAP", videoap_result))
    return videoap_result

def videpAP_005_045(gt, submit_dir):
    ap = 0
    for i in range(9):
        th = 0.05 + 0.05 * i
        new_gt=deepcopy(gt)
        ap += videoAP(new_gt, submit_dir, th, print_info=False)
    ap = ap / 9
    return ap

def videpAP_050_095(gt, submit_dir):
    ap = 0
    for i in range(10):
        th = 0.5 + 0.05 * i
        new_gt=deepcopy(gt)
        ap += videoAP(new_gt, submit_dir, th, print_info=False)
    ap = ap / 10
    return ap

def videpAP_010_090(gt, submit_dir):
    ap = 0
    for i in range(9):
        th = 0.1 + 0.1 * i
        new_gt=deepcopy(gt)
        ap += videoAP(new_gt, submit_dir, th, print_info=False)
    ap = ap / 9
    return ap

# input_dir='/Users/liyix/Downloads/'
# output_dir='.'

input_dir = sys.argv[1]
output_dir = sys.argv[2]

submit_dir = os.path.join(input_dir, 'res')
truth_dir = os.path.join(input_dir, 'ref')

gt=pickle.load(open(os.path.join(truth_dir,'multisports_half_test_GT.pkl'),'rb'))

new_gt=deepcopy(gt)

F=frameAP(new_gt,submit_dir, 0.5 ,print_info=False)

print(F)

new_gt=deepcopy(gt)

V1=videoAP(new_gt,submit_dir, 0.2,print_info=False)

print(V1)

new_gt=deepcopy(gt)

V11=videoAP(new_gt,submit_dir, 0.5,print_info=False)

print(V11)

new_gt=deepcopy(gt)

V2=videpAP_005_045(new_gt,submit_dir)

print(V2)

new_gt=deepcopy(gt)

V3=videpAP_050_095(new_gt,submit_dir)

print(V3)

new_gt=deepcopy(gt)

V4=videpAP_010_090(new_gt,submit_dir)

print(V4)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
with open(os.path.join(output_dir, 'scores.txt'), 'w') as output_file:
#    print(1)
    output_file.write("F@0.5:{0}\n".format(F))
#    print(1)
    output_file.write("V@0.2:{0}\n".format(V1))
#    print(1)
    output_file.write("V@0.5:{0}\n".format(V11))
#    print(1)
    output_file.write("V@0.05-0.45:{0}\n".format(V2))
#    print(1)
    output_file.write("V@0.50-0.95:{0}\n".format(V3))
#    print(1)
    output_file.write("V@0.10-0.90:{0}\n".format(V4))
#    print(1)
output_file.close()
