%get dataset info
function datasetInfo=getSODatasetInfo(datasetName)
    rootPath='/disk1/nianliu/Data/Salient_Object/Benchmarks/';
	resultsPath=['../img_mask/results/' datasetName];
	flag=1;
    switch datasetName
        case 'BSD'
            rootPath=[rootPath datasetName];
            imgPath=[rootPath '/images'];
            maskPath=[rootPath '/GT'];
            allFiles=dir(imgPath);
            [~,im_ext]=strtok(allFiles(3).name,'.');
            imgFiles=dir([imgPath '/*' im_ext]);
            maskFiles=dir([maskPath '/*.png']);
        case 'DUT-O'
            rootPath=[rootPath datasetName];
            imgPath=[rootPath '/images'];
            maskPath=[rootPath '/GT'];
            allFiles=dir(imgPath);
            [~,im_ext]=strtok(allFiles(3).name,'.');
            imgFiles=dir([imgPath '/*' im_ext]);
            maskFiles=dir([maskPath '/*.png']);
        case 'ECSSD'
            rootPath=[rootPath datasetName];
            imgPath=[rootPath '/images'];
            maskPath=[rootPath '/GT'];
            allFiles=dir(imgPath);
            [~,im_ext]=strtok(allFiles(3).name,'.');
            imgFiles=dir([imgPath '/*' im_ext]);
            maskFiles=dir([maskPath '/*.png']);
        case 'MSRA10K'
            rootPath=[rootPath datasetName '/MSRA10K_Imgs_GT'];
            imgPath=[rootPath '/images'];
            maskPath=[rootPath '/GT'];
            allFiles=dir(imgPath);
            [~,im_ext]=strtok(allFiles(3).name,'.');
            imgFiles=dir([imgPath '/*' im_ext]);
            maskFiles=dir([maskPath '/*.png']);
        case 'PASCAL1500'
            rootPath=[rootPath datasetName '/PASCAL1500_dataset'];
            imgPath=[rootPath '/images'];
            maskPath=[rootPath '/GT'];
            allFiles=dir(imgPath);
            [~,im_ext]=strtok(allFiles(3).name,'.');
            imgFiles=dir([imgPath '/*' im_ext]);
            maskFiles=dir([maskPath '/*.png']);
        case 'SED'
            rootPath=[rootPath datasetName '/SED'];
            imgPath=[rootPath '/images'];
            maskPath=[rootPath '/GT'];
            allFiles=dir(imgPath);
            [~,im_ext]=strtok(allFiles(3).name,'.');
            imgFiles=dir([imgPath '/*' im_ext]);
            maskFiles=dir([maskPath '/*.png']);
        case 'PASCAL-S'
            rootPath=[rootPath datasetName];
            imgPath=[rootPath '/images'];
            maskPath=[rootPath '/GT'];
            allFiles=dir(imgPath);
            [~,im_ext]=strtok(allFiles(3).name,'.');
            imgFiles=dir([imgPath '/*' im_ext]);
            maskFiles=dir([maskPath '/*.png']);
        case 'MSRA10K-test'
            rootPath=['/home/nianliu/Research/CNN-SO/img_mask/MSRA10K/test'];
            imgPath=[rootPath '/img'];
            maskPath=[rootPath '/mask'];
            allFiles=dir(imgPath);
            [~,im_ext]=strtok(allFiles(3).name,'.');
            imgFiles=dir([imgPath '/*' im_ext]);
            maskFiles=dir([maskPath '/*.png']);
        case 'DUT-O-test'
            rootPath=['/home/nianliu/Research/CNN-SO/img_mask/DUT-O/test'];
            imgPath=[rootPath '/img'];
            maskPath=[rootPath '/mask'];
            allFiles=dir(imgPath);
            [~,im_ext]=strtok(allFiles(3).name,'.');
            imgFiles=dir([imgPath '/*' im_ext]);
            maskFiles=dir([maskPath '/*.png']);
        case 'HKU-IS'
            rootPath=[rootPath datasetName];
            imgPath=[rootPath '/images'];
            maskPath=[rootPath '/GT'];
            allFiles=dir(imgPath);
            [~,im_ext]=strtok(allFiles(3).name,'.');
            imgFiles=dir([imgPath '/*' im_ext]);
            maskFiles=dir([maskPath '/*.png']);
        case 'DUTS-TR'
            rootPath=[rootPath '/DUTS/' datasetName];
            imgPath=[rootPath '/' datasetName '-Image'];
            maskPath=[rootPath '/' datasetName '-Mask'];
            allFiles=dir(imgPath);
            [~,im_ext]=strtok(allFiles(3).name,'.');
            imgFiles=dir([imgPath '/*' im_ext]);
            maskFiles=dir([maskPath '/*.png']);
        case 'DUTS-TE'
            rootPath=[rootPath '/DUTS/' datasetName];
            imgPath=[rootPath '/' datasetName '-Image'];
            maskPath=[rootPath '/' datasetName '-Mask'];
            allFiles=dir(imgPath);
            [~,im_ext]=strtok(allFiles(3).name,'.');
            imgFiles=dir([imgPath '/*' im_ext]);
            maskFiles=dir([maskPath '/*.png']);
        otherwise
		    rootPath=[];
			maskPath=[];
			maskFiles=[];
		    imgPath=datasetName;
			allFiles=dir(imgPath);
			[~,im_ext]=strtok(allFiles(3).name,'.');
            imgFiles=dir([imgPath '/*' im_ext]);
			resultsPath=[datasetName '/saliency_maps'];
			mkdir(resultsPath);
            flag=0;
    end
    if flag && length(imgFiles)~=length(maskFiles)
        disp('Error! Mask map num does''t equal to image num!');
        return
    end
    datasetInfo.rootPath=rootPath;
    datasetInfo.imgPath=imgPath;
    datasetInfo.maskPath=maskPath;
    datasetInfo.imgFiles=imgFiles;
    datasetInfo.maskFiles=maskFiles;
    datasetInfo.imgNum=length(imgFiles);
    datasetInfo.resultsPath=resultsPath;
end