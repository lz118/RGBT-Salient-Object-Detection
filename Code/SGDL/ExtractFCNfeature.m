function [meanVgg1,meanVgg2] = ExtractFCNfeature(matRoot,imname,pixelList,m,n)

layer1 = 32;
layer2 = 6;

padding = [0 100 100 100 100 ... 
            52 52 52 52 52 ... 
            26 26 26 26 26 26 26 ... 
            14 14 14 14 14 14 14 ...
            7 7 7 7 7 7 7 ...
            4];


matName1 = [matRoot,'32','\',imname,'.mat'];
eval(['load ',matName1,';']);
temp = layer32;    % feat

vgg_feat1 = temp(padding(layer1):end-padding(layer1)+1,padding(layer1):end-padding(layer1)+1,:);
vgg_feat1 = double(imresize(vgg_feat1,[m,n]));
meanVgg1 = GetMeanColor(vgg_feat1,pixelList,'vgg'); 
matName2 = [matRoot,'6','\',imname,'.mat'];
eval(['load ',matName2,';']);
temp = layer6;

vgg_feat2 = temp(padding(layer2):end-padding(layer2)+1,padding(layer2):end-padding(layer2)+1,:);
vgg_feat2 = double(imresize(vgg_feat2,[m,n]));
meanVgg2 = GetMeanColor(vgg_feat2,pixelList,'vgg');