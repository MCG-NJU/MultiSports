import os
import pickle
import numpy as np
from copy import deepcopy
import math

"""VideomAP_error is based on the code from ACT. https://github.com/vkalogeiton/caffe/blob/act-detector/act-detector-scripts/ACT.py"""


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


def iou3dt(b1, b2, spatialonly=False, temporalonly=False):
    """Compute the spatio-temporal IoU between two tubes"""

    tmin = max(b1[0, 0], b2[0, 0])
    tmax = min(b1[-1, 0], b2[-1, 0])

    if tmax < tmin:
        return 0.0

    temporal_inter = tmax - tmin + 1
    temporal_union = max(b1[-1, 0], b2[-1, 0]) - min(b1[0, 0], b2[0, 0]) + 1

    tube1 = b1[int(np.where(b1[:, 0] == tmin)[0]): int(np.where(b1[:, 0] == tmax)[0]) + 1, :]
    tube2 = b2[int(np.where(b2[:, 0] == tmin)[0]): int(np.where(b2[:, 0] == tmax)[0]) + 1, :]

    if temporalonly:
        return temporal_inter / temporal_union
    return iou3d(tube1, tube2) * (1. if spatialonly else temporal_inter / temporal_union)

def pr_to_ap(pr):
    """Compute AP given precision-recall
    pr is a Nx2 array with first row being precision and second row being recall
    """

    prdif = pr[1:, 1] - pr[:-1, 1]
    prsum = pr[1:, 0] + pr[:-1, 0]

    return np.sum(prdif * prsum * 0.5)

def videoAP_error(prediction_file, GT, threshold=0.1, redo=True):

    th=threshold
    th_s = math.sqrt(th)
    th_t = math.sqrt(th)

    eval_file = os.path.join("videoAP{:g}_error.pkl".format(th))
    print('th is', th)
    print('th_s is', th_s)
    print('th_t is', th_t)

    if os.path.isfile(eval_file) and not redo:
        with open(eval_file, 'rb') as fid:
            res = pickle.load(fid)
    else:
        vlist = GT['test_videos'][0]
        ## load detected tubes
        alldets=pickle.load(open(prediction_file,'rb'))

        res = {}
        ##compute video error for every class
        for ilabel in range(len(GT['labels'])):
            if GT['labels'][ilabel] in ['aerobic kick jump', 'aerobic off axis jump', 'aerobic butterfly jump', 'aerobic balance turn','basketball save','basketball jump ball']:
                print('do not evaluate {}'.format(GT['labels'][ilabel]))
                continue
            detections = alldets[ilabel]
            gt={}
            ## load GT
            for v in vlist:
                gt[v]=deepcopy(GT['gttubes'][v])
            dupgt = deepcopy(gt)

            pr = np.zeros((len(detections) + 1, 11), dtype=np.float32)
            pr[0, 0] = 1.0
            pr[0, 1] = 0.0

            fn=0
            for v in dupgt:
                if ilabel in dupgt[v]:
                    fn=fn+len(dupgt[v][ilabel])
            fp = 0  # false positives
            tp = 0  # true positives
            ER = 0  # repeat error repeat predict for the same instance
            EN = 0  # extra error
            EL = 0  # localization errors
            EC = 0  # classification error
            ET = 0  # timing error
            ECT = 0 # cls + time
            ECL = 0 # cls + loc
            ETL = 0 # time + loc
            ECTL = 0 # cls + time + loc
            
            for i, j in enumerate(np.argsort(-np.array([dd[1] for dd in detections]))):
                v, score, tube = detections[j]
                ispositive = False
                end = False
                if ilabel in gt[v]:
                    ious = [iou3dt(g, tube) for g in gt[v][ilabel]]
                    amax = np.argmax(ious)
                    if ious[amax] >= th:
                        ispositive = True
                        del gt[v][ilabel][amax]
                        if len(gt[v][ilabel])==0:
                            del gt[v][ilabel]
                        end=True
                if ilabel in dupgt[v] and end == False:
                    ious = [iou3dt(g, tube) for g in dupgt[v][ilabel]]
                    amax = np.argmax(ious)
                    if ious[amax] >= th:
                        ER += 1
                        end=True
                if end==False:
                    ious=[]
                    for ll in dupgt[v]:
                        if ll==ilabel:
                            continue
                        for g in dupgt[v][ll]:
                            ious.append(iou3dt(g, tube))
                    if ious!=[]:
                        amax = np.argmax(ious)
                        if ious[amax] >= th:
                            EC += 1
                            end=True
                if end == False:
                    all_gt=[]
                    ious=[]
                    for ll in dupgt[v]:
                        for g in dupgt[v][ll]:
                            all_gt.append((ll,g))
                            ious.append(iou3dt(g, tube))
                    amax = np.argmax(ious)
                    assert(ious[amax]<th)
                    if ious[amax]>0:
                        t_iou=iou3dt(all_gt[amax][1], tube, temporalonly=True)
                        s_iou=iou3dt(all_gt[amax][1], tube, spatialonly=True)
                        if all_gt[amax][0]==ilabel:
                            assert(t_iou<th_t or s_iou<th_s)
                            if t_iou >= th_t:
                                EL+=1
                                end=True
                            elif s_iou >= th_s:
                                ET+=1
                                end=True
                            else:
                                ETL+=1
                                end=True
                        else:
                            assert(t_iou<th_t or s_iou<th_s)
                            if t_iou >= th_t:
                                ECL+=1
                                end=True
                            elif s_iou >= th_s:
                                ECT+=1
                                end=True
                            else:
                                ECTL+=1
                                end=True
                    else:
                        EN += 1
                        end = True
                assert(end == True) 
                if ispositive:
                    tp += 1
                    fn -= 1
                else:
                    fp += 1
                pr[i + 1, 0] = max(float(tp) / float(tp + fp), 0.)
                pr[i + 1, 1] = max(float(tp) / float(tp + fn), 0.)
                pr[i + 1, 2] = max(float(ER) / float(tp + fp), 0.)
                pr[i + 1, 3] = max(float(EN) / float(tp + fp), 0.)
                pr[i + 1, 4] = max(float(EL) / float(tp + fp), 0.)
                pr[i + 1, 5] = max(float(EC) / float(tp + fp), 0.)
                pr[i + 1, 6] = max(float(ET) / float(tp + fp), 0.)
                pr[i + 1, 7] = max(float(ECT) / float(tp + fp), 0.)
                pr[i + 1, 8] = max(float(ECL) / float(tp + fp), 0.)
                pr[i + 1, 9] = max(float(ETL) / float(tp + fp), 0.)
                pr[i + 1, 10] = max(float(ECTL) / float(tp + fp), 0.)
            
            res[GT['labels'][ilabel]] = pr

        # save results
        with open(eval_file, 'wb') as fid:
            pickle.dump(res, fid)

    # display results
    AP = 100 * np.array([pr_to_ap(res[label][:, [0, 1]]) for label in res])
    othersap = [100 * np.array([pr_to_ap(res[label][:, [j, 1]]) for label in res]) for j in range(2, 11)]

    ER = othersap[0]
    EN = othersap[1]
    EL = othersap[2]
    EC = othersap[3]
    ET = othersap[4]
    ECT = othersap[5]
    ECL = othersap[6]
    ETL = othersap[7]
    ECTL = othersap[8]
    #missed detections = 1-recalll
    EM = 100 - 100 * np.array([res[label][-1, 1] for label in res])

    LIST = [AP, ER, EN, EL, EC, ET, ECT, ECL, ETL, ECTL, EM]

    print('Error Analysis')

    print("")
    print("{:20s} {:8s} {:8s} {:8s} {:8s} {:8s} {:8s} {:8s} {:8s} {:8s} {:8s} {:8s}".format('label', '  AP ', '  Repeat ', ' Extra ', ' Loc. ', ' Cls. ', ' Time ', ' Cls.+Time ',' Cls.+Loc. ',' Time+Loc. ',' C+T+L ', ' missed '))
    print("")
    for il, label in enumerate(res):
        print("{:20s} ".format(label) + " ".join(["{:8.2f}".format(L[il]) for L in LIST]))
    print("")
    print("{:20s} ".format("mean") + " ".join(["{:8.2f}".format(np.mean(L)) for L in LIST]))
    print("")


if __name__ == "__main__":
    prediction_file='video_detections.pkl'
    gt_file='multisports_GT.pkl'
    GT=pickle.load(open(gt_file,'rb'))
    videoAP_error(prediction_file, GT)
