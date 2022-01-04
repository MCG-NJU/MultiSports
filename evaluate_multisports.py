import pickle
import numpy as np
import argparse
import math

def area2d_voc(b):
    """Compute the areas for a set of 2D boxes"""
    return (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

def overlap2d_voc(b1, b2):
    """Compute the overlaps between a set of boxes b1 and one box b2"""
    xmin = np.maximum(b1[:, 0], b2[:, 0])
    ymin = np.maximum(b1[:, 1], b2[:, 1])
    xmax = np.minimum(b1[:, 2], b2[:, 2])
    ymax = np.minimum(b1[:, 3], b2[:, 3])

    width = np.maximum(0, xmax - xmin)
    height = np.maximum(0, ymax - ymin)

    return width * height

def iou2d_voc(b1, b2):
    """Compute the IoU between a set of boxes b1 and 1 box b2"""
    if b1.ndim == 1:
        b1 = b1[None, :]
    if b2.ndim == 1:
        b2 = b2[None, :]

    assert b2.shape[0] == 1

    ov = overlap2d_voc(b1, b2)

    return ov / (area2d_voc(b1) + area2d_voc(b2) - ov)

def iou3d_voc(b1, b2):
    """Compute the IoU between two tubes with same temporal extent"""
    assert b1.shape[0] == b2.shape[0]
    assert np.all(b1[:, 0] == b2[:, 0])

    ov = overlap2d_voc(b1[:, 1:5], b2[:, 1:5])

    return np.mean(ov / (area2d_voc(b1[:, 1:5]) + area2d_voc(b2[:, 1:5]) - ov))

def iou3dt_voc(b1, b2, spatialonly=False, temporalonly=False):
    """Compute the spatio-temporal IoU between two tubes"""
    tmin = max(b1[0, 0], b2[0, 0])
    tmax = min(b1[-1, 0], b2[-1, 0])

    if tmax < tmin:
        return 0.0

    temporal_inter = tmax - tmin
    temporal_union = max(b1[-1, 0], b2[-1, 0]) - min(b1[0, 0], b2[0, 0])

    tube1 = b1[int(np.where(b1[:, 0] == tmin)[0]): int(np.where(b1[:, 0] == tmax)[0]) + 1, :]
    tube2 = b2[int(np.where(b2[:, 0] == tmin)[0]): int(np.where(b2[:, 0] == tmax)[0]) + 1, :]

    if temporalonly:
        return temporal_inter / temporal_union
    return iou3d_voc(tube1, tube2) * (1. if spatialonly else temporal_inter / temporal_union)

def pr_to_ap_voc(pr):
    precision = pr[:,0]
    recall = pr[:,1]
    recall = np.concatenate([[0], recall, [1]])
    precision = np.concatenate([[0], precision, [0]])

    # Preprocess precision to be a non-decreasing array
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = np.maximum(precision[i], precision[i + 1])

    indices = np.where(recall[1:] != recall[:-1])[0] + 1
    average_precision = np.sum(
        (recall[indices] - recall[indices - 1]) * precision[indices])
    return average_precision

def frameAP(groundtruth, detections, thr, print_info=True):

    GT=pickle.load(open(groundtruth,'rb'))
    vlist = GT['test_videos'][0]
    alldets=pickle.load(open(detections,'rb'))

    results = {}
    for ilabel, label in enumerate(GT['labels']):
        # detections of this class
        if label in ['aerobic kick jump', 'aerobic off axis jump', 'aerobic butterfly jump', 'aerobic balance turn','basketball save','basketball jump ball']:
            if print_info:
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
        pr = np.empty((detections.shape[0], 2), dtype=np.float64)  # precision,recall
        gt_num = sum([g.shape[0] for g in gt.values()])
        if gt_num==0:
            if print_info:
                print('no such label',ilabel,label)
            continue
        fp = 0  # false positives
        tp = 0  # true positives

        is_gt_box_detected={}
        for i, j in enumerate(np.argsort(-detections[:, 3])):
            k = (int(detections[j, 0]), int(detections[j, 1]))
            box = detections[j, 4:8]
            ispositive = False
            
            if k in gt:
                if k not in is_gt_box_detected:
                    is_gt_box_detected[k] = np.zeros(gt[k].shape[0], dtype=bool)
                ious = iou2d_voc(gt[k], box)
                amax = np.argmax(ious)

                if ious[amax] >= thr:
                    if not is_gt_box_detected[k][amax]:
                        ispositive = True
                        is_gt_box_detected[k][amax]=True

            if ispositive:
                tp += 1
            else:
                fp += 1

            pr[i, 0] = float(tp) / float(tp + fp)
            pr[i, 1] = float(tp) / float(gt_num)

        results[label] = pr

    # display results
    ap = 100 * np.array([pr_to_ap_voc(results[label]) for label in results])
    class_result={}
    for label in results:
        class_result[label]=pr_to_ap_voc(results[label])*100
    frameap_result = np.mean(ap)
    if print_info:
        print('frameAP_{}\n'.format(thr))
        for l in class_result:
            print("{:20s} {:8.2f}".format(l,class_result[l]))
        print("{:20s} {:8.2f}".format("mAP", frameap_result))
    return frameap_result

def videoAP(groundtruth, detections, thr, print_info=True):

    GT=pickle.load(open(groundtruth,'rb'))
    vlist = GT['test_videos'][0]
    alldets=pickle.load(open(detections,'rb'))

    res = {}
    for ilabel in range(len(GT['labels'])):
        if GT['labels'][ilabel] in ['aerobic kick jump', 'aerobic off axis jump', 'aerobic butterfly jump', 'aerobic balance turn','basketball save','basketball jump ball']:
            if print_info:
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
        pr = np.empty((len(detections), 2), dtype=np.float64)

        gt_num = sum([len(g) for g in gt.values()])  # false negatives
        fp = 0  # false positives
        tp = 0  # true positives
        if gt_num==0:
            if print_info:
                print('no such label', ilabel, GT['labels'][ilabel])
            continue
        is_gt_box_detected={}
        for i, j in enumerate(np.argsort(-np.array([dd[1] for dd in detections]))):
            v, score, tube = detections[j]
            ispositive = False
            if v in gt:
                if v not in is_gt_box_detected:
                    is_gt_box_detected[v] = np.zeros(len(gt[v]), dtype=bool)
                ious = [iou3dt_voc(g, tube) for g in gt[v]]
                amax = np.argmax(ious)
                if ious[amax] >= thr:
                    if not is_gt_box_detected[v][amax]:
                        ispositive = True
                        is_gt_box_detected[v][amax] = True

            if ispositive:
                tp += 1
            else:
                fp += 1

            pr[i, 0] = float(tp) / float(tp + fp)
            pr[i, 1] = float(tp) / float(gt_num)
        res[GT['labels'][ilabel]] = pr

    # display results
    ap = 100 * np.array([pr_to_ap_voc(res[label]) for label in res])
    videoap_result = np.mean(ap)
    class_result={}
    for label in res:
        class_result[label]=pr_to_ap_voc(res[label])*100
    if print_info:
        print('VideoAP_{}\n'.format(thr))
        for l in class_result:
            print("{:20s} {:8.2f}".format(l,class_result[l]))
        print("{:20s} {:8.2f}".format("mAP", videoap_result))
    return videoap_result

def videoAP_all(groundtruth, detections):
    high_ap = 0
    for i in range(10):
        thr = 0.5 + 0.05 * i
        high_ap += videoAP(groundtruth, detections, thr, print_info=False)
    high_ap = high_ap / 10.0
    
    low_ap=0
    for i in range(9):
        thr = 0.05 + 0.05 * i
        low_ap += videoAP(groundtruth, detections, thr, print_info=False)
    low_ap = low_ap / 9.0

    all_ap=0
    for i in range(9):
        thr = 0.1+0.1*i
        all_ap += videoAP(groundtruth, detections, thr, print_info=False)
    all_ap= all_ap/9.0

    print('\nVideoAP_0.05:0.45: {:8.2f} \n'.format(low_ap))
    print('VideoAP_0.10:0.90: {:8.2f} \n'.format(all_ap))
    print('VideoAP_0.50:0.95: {:8.2f} \n'.format(high_ap))

def videoAP_error(groundtruth, detections, thr):

    GT=pickle.load(open(groundtruth,'rb'))
    vlist = GT['test_videos'][0]
    alldets=pickle.load(open(detections,'rb'))
    
    th_s = math.sqrt(thr)
    th_t = math.sqrt(thr)

    print('th is', thr)
    print('th_s is', th_s)
    print('th_t is', th_t)

    res = {}
    dupgt={}
    for v in vlist:
        dupgt[v]=GT['gttubes'][v]
    ##compute video error for every class
    for ilabel in range(len(GT['labels'])):
        if GT['labels'][ilabel] in ['aerobic kick jump', 'aerobic off axis jump', 'aerobic butterfly jump', 'aerobic balance turn','basketball save','basketball jump ball']:
            print('do not evaluate {}'.format(GT['labels'][ilabel]))
            continue
        detections = alldets[ilabel]

        pr = np.zeros((len(detections), 11), dtype=np.float32)

        gt_num=0
        for v in dupgt:
            if ilabel in dupgt[v]:
                gt_num=gt_num+len(dupgt[v][ilabel])
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
        
        is_gt_box_detected={}
        for i, j in enumerate(np.argsort(-np.array([dd[1] for dd in detections]))):
            v, score, tube = detections[j]
            ispositive = False
            end = False
            if ilabel in dupgt[v]:
                if v not in is_gt_box_detected:
                    is_gt_box_detected[v] = np.zeros(len(dupgt[v][ilabel]), dtype=bool)
                ious = [iou3dt_voc(g, tube) for g in dupgt[v][ilabel]]
                amax = np.argmax(ious)
                if ious[amax] >= thr:
                    if not is_gt_box_detected[v][amax]:
                        ispositive=True
                        is_gt_box_detected[v][amax]=True
                    else:
                        ER += 1
                    end=True
            if end==False:
                ious=[]
                for ll in dupgt[v]:
                    if ll==ilabel:
                        continue
                    for g in dupgt[v][ll]:
                        ious.append(iou3dt_voc(g, tube))
                if ious!=[]:
                    amax = np.argmax(ious)
                    if ious[amax] >= thr:
                        EC += 1
                        end=True
            if end == False:
                all_gt=[]
                ious=[]
                for ll in dupgt[v]:
                    for g in dupgt[v][ll]:
                        all_gt.append((ll,g))
                        ious.append(iou3dt_voc(g, tube))
                amax = np.argmax(ious)
                assert(ious[amax]<thr)
                if ious[amax]>0:
                    t_iou=iou3dt_voc(all_gt[amax][1], tube, temporalonly=True)
                    s_iou=iou3dt_voc(all_gt[amax][1], tube, spatialonly=True)
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
                # fn -= 1
            else:
                fp += 1
            assert(fp==(ER+EN+EL+EC+ET+ECT+ECL+ETL+ECTL))
            pr[i, 0] = max(float(tp) / float(tp + fp), 0.)
            pr[i, 1] = max(float(tp) / float(gt_num), 0.)
            pr[i, 2] = max(float(ER) / float(tp + fp), 0.)
            pr[i, 3] = max(float(EN) / float(tp + fp), 0.)
            pr[i, 4] = max(float(EL) / float(tp + fp), 0.)
            pr[i, 5] = max(float(EC) / float(tp + fp), 0.)
            pr[i, 6] = max(float(ET) / float(tp + fp), 0.)
            pr[i, 7] = max(float(ECT) / float(tp + fp), 0.)
            pr[i, 8] = max(float(ECL) / float(tp + fp), 0.)
            pr[i, 9] = max(float(ETL) / float(tp + fp), 0.)
            pr[i, 10] = max(float(ECTL) / float(tp + fp), 0.)
            
        res[GT['labels'][ilabel]] = pr

    # display results
    AP = 100 * np.array([pr_to_ap_voc(res[label][:, [0, 1]]) for label in res])
    othersap = [100 * np.array([pr_to_ap_voc(res[label][:, [j, 1]]) for label in res]) for j in range(2, 11)]

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
    EM=[]
    for label in res:
        if res[label].shape[0]!=0:
            EM.append(100-100*res[label][-1, 1])
        else:
            EM.append(100)
    EM=np.array(EM)

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
    parser = argparse.ArgumentParser()
    parser.add_argument(
      "task",
      help="evaluate type.")
    parser.add_argument(
      "groundtruth",
      help="pkl file containing ground truth.")
    parser.add_argument(
      "detections",
      help="pkl file containing detections.")
    parser.add_argument(
      "-t",
      "--thr",
      help="threshold evaluating detections.",
      type=float,
      default=0.5)
    args=parser.parse_args()
    if args.task == 'frameAP':
        frameAP(args.groundtruth, args.detections, args.thr)
    elif args.task == 'videoAP':
        videoAP(args.groundtruth, args.detections, args.thr)
    elif args.task == 'videoAP_all':
        videoAP_all(args.groundtruth, args.detections)
    elif args.task == 'videoAP_error':
        videoAP_error(args.groundtruth, args.detections, args.thr)
    else:
        raise NotImplementedError('Not implemented:' + args.task)
