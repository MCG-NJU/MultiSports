# README
We will release our baseline codes and models soon.

Paper: [MultiSports: A Multi-Person Video Dataset of Spatio-Temporally Localized Sports Actions](https://arxiv.org/pdf/2105.07404.pdf)

Homepage (Download): [DeeperAction/MultiSports](https://deeperaction.github.io/multisports/)

## Download Data and Annotations

Please register on our [competition page](https://competitions.codalab.org/competitions/32066), and download data and annotations in the Participate/Data part.

## Download Person Boxes

We provide the **person boxes** generated by the person detector of Faster R-CNN with a ResNeXt-101-FPN backbone in the links below. We finetune the person detector only with the train set data.

Onedrive: https://1drv.ms/f/s!AtjeLq7YnYGRe3eQMuQk5GYYu40

Baidu Wangpan: https://pan.baidu.com/s/1zOylA-idz2foeEaU1gx6sw (password: 5ccx)

The pkl file has the below data structure:

{"video_name,frame_number":boxes}

video_name: str

frame_number: int (starts from 1)

boxes: a numpy array with n rows and 5 columns, \<x1\> \<y1\> \<x2\> \<y2\> \<score\>. x1, x2, y1, y2 are normalized with respect to frame size, which are between 0.0-1.0. We only save the box with score higher than 0.05.

## Evaluation Tools

**evaluate.py** is our evaluation code, which is modified from ACT(https://github.com/vkalogeiton/caffe/blob/act-detector/act-detector-scripts/ACT.py). You can register on [this website](https://competitions.codalab.org/competitions/33355) and submit predictions for test set evaluation in the Participate/Submit part.

**VideomAP_error.py** is our video mAP error analysis code, which is based on the frame mAP error analysis code of ACT(https://github.com/vkalogeiton/caffe/blob/act-detector/act-detector-scripts/ACT.py).

## Prediction Example

**submissions.zip** is our example submission file on 50% test set.


If you find our code or paper useful, please cite as
```
@article{li2021multisports,
  title={MultiSports: A Multi-Person Video Dataset of Spatio-Temporally Localized Sports Actions},
  author={Li, Yixuan and Chen, Lei and He, Runyu and Wang, Zhenzhi and Wu, Gangshan and Wang, Limin},
  journal={arXiv preprint arXiv:2105.07404},
  year={2021}
}
```
