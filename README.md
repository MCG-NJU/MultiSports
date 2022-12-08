# README
We will release our baseline codes and models soon.

Paper: [MultiSports: A Multi-Person Video Dataset of Spatio-Temporally Localized Sports Actions](https://arxiv.org/pdf/2105.07404.pdf)

Homepage: [DeeperAction/MultiSports](https://deeperaction.github.io/datasets/multisports.html)

## Download Data and Annotations

Please register on our [competition page](https://codalab.lisn.upsaclay.fr/competitions/3736), and download data and annotations in the Participate/Data part.

MultiSports Dataset License: [CC BY_NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)

## Download Person Boxes

We provide the **person boxes** generated by the person detector of Faster R-CNN with a ResNeXt-101-FPN backbone in the links below. We finetune the person detector only with the train set data.

Onedrive: https://1drv.ms/f/s!AtjeLq7YnYGRe3eQMuQk5GYYu40

Baidu Wangpan: https://pan.baidu.com/s/1zOylA-idz2foeEaU1gx6sw (password: 5ccx)

The pkl file has the below data structure:
```
{"video_name, frame_number":boxes}

video_name: str

frame_number: int (starts from 1)
```
boxes: a numpy array with n rows and 5 columns, \<x1\> \<y1\> \<x2\> \<y2\> \<score\>. x1, x2, y1, y2 are normalized with respect to frame size, which are between 0.0-1.0. We only save the box with score higher than 0.05.

## Evaluation Tools

**evaluate_multisports.py** is our evaluation code, which is modified from ACT(https://github.com/vkalogeiton/caffe/blob/act-detector/act-detector-scripts/ACT.py). We change the calculation of PR (Precision-Recall) area same with PASCAL VOC.
```
Usage:
python3 evaluate_multisports.py task gt_file detection_file -t thr

evaluate frameAP:
python3 evaluate_multisports.py frameAP multisports_GT.pkl frame_detections.pkl
evaluate videoAP:
python3 evaluate_multisports.py videoAP multisports_GT.pkl video_detections.pkl -t 0.2
evaluate videoAP with different thresholds:
python3 evaluate_multisports.py videoAP_all multisports_GT.pkl video_detections.pkl
analyze videoAP error:
python3 evaluate_multisports.py videoAP_error multisports_GT.pkl video_detections.pkl -t 0.2
```
**frame_detections.pkl** is a list. Every item is a numpy array with shape (8,) , numpy.array([video_index, frame_number, label_index, score, x1, y1, x2, y2]). video_index is the index of the video in the multisports_GT['test_videos'][0], which starts from 0. For example, 0 is the index of 'aerobic_gymnastics/v_crsi07chcV8_c004' in multisports_GT['test_videos'][0]. frame_number starts from 1. label_index starts from 0. score is the score of this box, which affects the frame mAP result.

**video_detections.pkl** is a dictionary that associates from each index of label (start from 0), a list of tubes. A tube is a tuple (tube_v, tube_score, tube_boxes). tube_v is the video name, such as 'aerobic_gymnastics/v_crsi07chcV8_c004'. tube_score is the score of this tube, which affects the video mAP result. tube_boxes is a numpy array with tube-length rows and 6 columns, every row is [frame number, x1, y1, x2, y2, box_score]. frame number starts from 1. box_score is the single frame's confidence and does not affect the video mAP result. 

We provide examples of frame_detections.pkl and video_detections.pkl of slowonly in [examples](https://github.com/MCG-NJU/MultiSports/tree/main/examples), whose evaluation results with different PR area calculation are shown below.

|  | frameAP@0.5 | videoAP@0.2 | videoAP@0.5 |
|---|---|---|:--|
| PR_ACT | 16.84 | 15.75 | 5.84 |
| PR_PASCAL_VOC | 17.03 | 15.86 | 5.88 |



<!-- ## Prediction Example

**submissions.zip** is our example submission file on 50% test set.
 -->

If you find our code or paper useful, please cite as
```
@InProceedings{Li_2021_ICCV,
    author    = {Li, Yixuan and Chen, Lei and He, Runyu and Wang, Zhenzhi and Wu, Gangshan and Wang, Limin},
    title     = {MultiSports: A Multi-Person Video Dataset of Spatio-Temporally Localized Sports Actions},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {13536-13545}
}
```
