# PiCANet

source code for our CVPR 2018 paper [PiCANet: Learning Pixel-wise Contextual Attention for Saliency Detection](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/1251.pdf) 
by Nian Liu, Junwei Han, and Ming-Hsuan Yang.

Created by Nian Liu, Email: liunian228@gmail.com

## Usage:
1. Cd to ./caffe, install our modified caffe (given in the source code) and its MATLAB wrapper. Plese refer to 
http://caffe.berkeleyvision.org/installation.html for caffe installation.
2. Download our trained models from [Google drive](https://drive.google.com/open?id=1sY1SLH-2KXZsVZ3rRYf--QMctXwAP1tQ). Unzip them to ./models/models.
3. Put your images into ./matlab/images.
4. Cd to ./matlab, run 'predict_SOs.m' and the saliency maps will be generated in ./matlab/results. You can also select whether to use the VGG based model or the ResNet50 based model in line 16 or 17.
5. You can also consider to use CRF post-processing to improve the detection results like we did in our paper. Please refer to Qibin Hou's [code](https://github.com/Andrew-Qibin/dss_crf).
6. We also provide our saliency maps [here](https://drive.google.com/open?id=1IXbgSp9g0mp0bN3yY137wzq6UW0Q6CaU).

## Training:
1. Download the pretrained VGG model ([vgg16_20M.caffemodel](http://liangchiehchen.com/projects/Init%20Models.html) from deeplab) or the [ResNet50](https://github.com/KaimingHe/deep-residual-networks) model. Modify the model directories in ./models/train_SO.sh.
2. Prepare your images, ground truth saliency maps, and the list file (please refer to ./matlab/list/train_list.txt). Modify corresponding contents in prototxt files.
3. Cd to ./models, run ```sh train_SO.sh``` to start training.

## Acknowledgement:
Our code uses some opensource code from [deeplab](https://bitbucket.org/aquariusjay/deeplab-public-ver2), [hybridnet](https://github.com/stephenyan1231/caffe-hybridnet), and a [caffe pull request](https://github.com/BVLC/caffe/pull/2016) to reduce GPU memory usage. Thank the authors.


## Citing our work
Please cite our work if it helps your research:
```
@inproceedings{liu2018picanet,
  title={PiCANet: Learning Pixel-wise Contextual Attention for Saliency Detection},
  author={Liu, Nian and Han, Junwei and Yang, Ming-Hsuan},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={3089--3098},
  year={2018}
}
```
