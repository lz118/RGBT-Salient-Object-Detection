% function feat = get_features_inception(im, cos_window, layers)
%GET_FEATURES
%   Extracts dense features from image.
%
addpath('mex')
img=imread('.\0001.jpg');
%     if ~exist('net','var')
%         inital_net;
%     end
    global res;
    global net1;   
%     
%     if isempty(net1),
%         initial_net_inception();
%     end
net1 = dagnn.DagNN.loadobj(load('.\FCN2015CVPR\pascal-fcn32s-dag.mat')) ;
%        net=vl_simplenn_move(net,'gpu');
%     sz_window=size(cos_window);
   %net= load('/home/waynecool/imagenet_vgg_verydeep_19.mat')
    img = single(img); % note: 255 range
%     img = imResampleMex(img, net.normalization.imageSize(1:2));
    img = imresize(img, [net1.meta.normalization.imageSize(1),net1.meta.normalization.imageSize(2)]);
%     img = img - net.normalization.averageImage;
tmp=single(zeros(500,500,3));
tmp(:,:,1)=net1.meta.normalization.averageImage(:,:,1);
tmp(:,:,2)=net1.meta.normalization.averageImage(:,:,2);
tmp(:,:,3)=net1.meta.normalization.averageImage(:,:,3);
img = img - tmp;
% img = gpuArray(img);
    
    % run the CNN
%     res=vl_simplenn(net,img);
  net1.eval({'data', img}) ; 
%   res=vl_simplenn(net,img);
    feat={};
    
   
