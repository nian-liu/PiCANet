%predict salient object saliency maps of a dataset
clear,clc
addpath('../caffe/matlab');
% addpath('/home/nianliu/Research/PiCANet_Saliency/caffe/matlab');

caffe.reset_all();
use_gpu=1;
gpu_id=3;
if use_gpu
  caffe.set_mode_gpu();
  caffe.set_device(gpu_id);
else
  caffe.set_mode_cpu();
end
modelDir='../models';
modelName='VGG_8s_75G432LP';
% modelName='Res50_8s_54G321LP';
model_config=[modelDir '/' modelName '_train_test.prototxt'];
model_file=[modelDir '/models/' modelName '_iter_20000.caffemodel'];

net = caffe.Net(model_config, model_file, 'test');

mean_pix = [104.008, 116.669, 122.675];
saveRootPath='./results';
mkdir(saveRootPath)
%scale={'5','4','3','2','1'};
scale={'1'};

%allDataset={'BSD','DUT-O-test','ECSSD','MSRA10K-test','PASCAL1500','SED','PASCAL-S'};
% allDataset={'BSD','DUT-O-test','SED','MSRA10K-test'};
% allDataset={'/home/nianliu/Research/CNN-SO/CVPR16_DHSNet_code/images'};
% allDataset={'BSD','DUT-O','ECSSD','SED','PASCAL-S','HKU-IS','DUTS-TE'};
allDataset={'./images'};
for datasetIdx=1:length(allDataset)
    datasetName=allDataset{datasetIdx};
    disp(datasetName);
    
    datasetInfo=getSODatasetInfo(datasetName);
    imgPath=datasetInfo.imgPath;
    %maskPath=datasetInfo.maskPath;
    imgFiles=datasetInfo.imgFiles;
    %maskFiles=datasetInfo.maskFiles;
    imgNum=datasetInfo.imgNum;
    resultsPath=[saveRootPath '/' modelName '/'];
    mkdir(resultsPath);
    
    for j=1:length(scale)
        if strcmp(scale{j},'1')
            smName{j}=modelName;
        else
            smName{j}=[modelName '_' scale{j}];
        end
        mkdir([resultsPath smName{j}]);
    end
    tic
    for i=1:imgNum
        disp(i)
        %close all
        
        image=imread([imgPath '/' imgFiles(i).name]);
        [imgName,~]=strtok(imgFiles(i).name,'.');

        im = single(image);
        im = imresize(im, [224, 224]);
        im = im(:, :, [3 2 1]);
        im = permute(im, [2 1 3]);
        for c = 1:3
            im(:, :, c) = im(:, :, c) - mean_pix(c);
        end

        net.blobs('img').set_data(im);
        net.forward_prefilled();
        
        for j=1:length(scale)
            sm=imresize((net.blobs(['sm_' scale{j}]).get_data())',[size(image,1),size(image,2)]);
            imwrite(sm, [resultsPath smName{j} '/' imgName '.png']);
        end
    end
    t=toc
end
caffe.reset_all();