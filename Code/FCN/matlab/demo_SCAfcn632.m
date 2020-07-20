% clc;
% clear;
% close all;
addpath('./others/');
addpath('./SLIC-DBSCAN/');
addpath('./added/');
% addpath('mex')
% global res;
% global net1;   
%%
% disp(['Initialization...']);
% tic;
% run vl_compilenn;
% run vl_setupnn;
% net1 = dagnn.DagNN.loadobj(load('E:\MyWorkOnSaliency\YaoQin\RunFCNFeat\matconvnet-1.0-beta19\Data\pascal-fcn32s-dag.mat')) ;
% 
% toc;
% disp(['Done!']);
%%
% SLIC algorithm parameters
spnumber = 200;  %the number of superpixels
% compactness=20;
% Single-layer Cellular Automata parameters
theta = 15;  % control the strength of similarity between neighbors
theta2 = 10;
a = 0.6;
b = 0.2;     % a and b control the strength of coherence

imgRoot = 'E:\Datasets\Database-S\PASCALS\PASCAL_S-Image\';% test image path
matRoot = ['E:\MyWorkOnSaliency\YaoQin\RunFCNFeat\fcnfeature_PASCALS\'];
% edgeRoot = ['E:\Datasets\Database-S\PASCALS\PASCALS_Edges\'];

% saldir1 = './vgg37pm_PASCALS/';
% saldir = ['.\vggfeat63612_PASCALS\'];
saldir = ['.\fcn632_PASCALS\'];
% mkdir(saldir);
% saldir2 = './use-gt-2layers-stage2_PASCALS/';
% saldir3 = './use-gt-2layers-stage3_PASCALS/';

% gtdir = 'E:\FengMY\MyDocument\Database-S\PASCALS\PASCALS-Mask\';
if ~exist(saldir, 'dir')
    mkdir(saldir);
end
% if ~exist(saldir2, 'dir')
%     mkdir(saldir2);
% end
% if ~exist(saldir3, 'dir')
%     mkdir(saldir3);
% end
% if ~exist(BDCON, 'dir')
%     mkdir(BDCON);
% end
% if ~exist(RES, 'dir')
%     mkdir(RES);
% end

imnames = dir([ imgRoot '*' 'jpg']);
%     load('res32.mat');
padding = [0 100 100 100 100 ... 
            52 52 52 52 52 ... 
            26 26 26 26 26 26 26 ... 
            14 14 14 14 14 14 14 ...
            7 7 7 7 7 7 7 ...
            4];
% layer = 1;
%% 2. Saliency Map Calculation
for ii = 1:length(imnames)
    disp(ii);
    imName = [ imgRoot imnames(ii).name ]; 
%     layer = 36;
    [input_im_,input_im,w]=removeframe(imName);% run a pre-processing to remove the image frame
    %%
%     img = single(input_im_); % note: 255 range
%     %     img = imResampleMex(img, net.normalization.imageSize(1:2));
%     img = imresize(img, [net1.meta.normalization.imageSize(1),net1.meta.normalization.imageSize(2)]);
%     %     img = img - net.normalization.averageImage;
%     tmp=single(zeros(500,500,3));
%     tmp(:,:,1)=net1.meta.normalization.averageImage(:,:,1);
%     tmp(:,:,2)=net1.meta.normalization.averageImage(:,:,2);
%     tmp(:,:,3)=net1.meta.normalization.averageImage(:,:,3);
%     img = img - tmp;
%     % img = gpuArray(img);
% 
%         % run the CNN
%     %     res=vl_simplenn(net,img);
%     disp(['Run FCN Feature...']);
%     tic;
%       net1.eval({'data', img}) ;
%     toc;
%     disp(['Done!']);
  %%
    input_imlab = RGB2Lab(input_im);
%     input_imlab = normalization(input_imlab, 0);
%     input_imlab = (input_imlab+120)/240;
%     imwrite(input_im,[saldir,imnames(ii).name(1:end-4),'.png']);
%     continue;
    [m,n,r]=size(input_im);
    
    %% Segment input rgb image into superpixels
%         [sulabel,impfactor8,pixelList] = SLIC_Split(input_im, spnumber);
%         IM_B = myDrawBounderis(sulabel,input_im);
%         imwrite(IM_B,[saldir,imnames(ii).name(1:end-4),'_suRGB.png']);
%         input_im = imresize(input_im,[14,14]);
        disp(['Run SLIC...']);
        tic;
        [sulabel,impfactor8,pixelList] = SLIC_Split(input_im, spnumber);
        toc;
        disp(['Done!']);
%         IM_B = myDrawBounderis(sulabel,input_im);
%         IM_B = imresize(IM_B,[224 224]);
%         imwrite(IM_B,[saldir,imnames(ii).name(1:end-4),'.png']);
%         continue;
%         sulabel = imresize(sulabel,[224 224],'nearest');
        supNum = max(sulabel(:));
%         pixelList = cell(supNum, 1);
%         for nn = 1:supNum
%             pixelList{nn} = find(sulabel == nn);
%         end
    %% get prior value for each superpixel
%     input_im_prior = ones(w(1),w(2))*0.5;
%     sal_prior=input_im_prior(w(3):w(4), w(5):w(6)); 
    S_prior=ones(supNum,1)*0.5; 
    %% get edges
%     im_edge=double(imread([edgeRoot,imnames(ii).name(1:end-4),'.png']));
% % %     input_im_prior=mat2gray(input_im_prior);
%     imedge=im_edge; % remove the image frame
% % %     [IM_B,bw] = myDrawBounderis(sulabel,input_im);
%     SupBound = findSupBoundaries(sulabel);
    
    %% get boundary superpixels

%     bdIds = extract_bg_sp(sulabel,m,n);
    bdIds = GetBndPatchIds(sulabel);
%     S_N1 = zeros(supNum,1);
%     S_N1(bdIds)=1;

    %% Get super-pixel properties
    
    meanRgbCol = GetMeanColor(input_im, pixelList,'rgb');
    meanLabCol = colorspace('Lab<-', double(meanRgbCol)/255);
    


    %     seg_vals = meanLabCol;
        % Color Name
    %     temp = load('w2crs');
    %     w2c = temp.w2crs;
    %     out_gray = get_feature_map(uint8(input_im*255), 'gray', w2c);
    %     out_CN = get_feature_map(uint8(input_im*255),'cn',w2c);
    %     im_cn = cat(3,out_gray,out_CN);
    %     meanCnCol = GetMeanColor(im_cn,pixelList);
    %     seg_vals = meanCnCol;
        % lbp
    %     [A,~] = LBP_uniform(rgb2gray(input_im));
    %      lbp_vals=zeros(supNum,1,59);
    %      STA=regionprops(superpixels,'all');
    %     for i=1:supNum
    %         temp=A(STA(i).PixelIdxList);
    %         lbp_vals(i,1,:)=hist(temp,1:59);
    %     end
    %     % lbp  59 D  
    %     lbp_vals=reshape(lbp_vals,spnum,59); 
    %%
%     impfactor = impfactor8;
%         edges8=[];
%         for i=1:supNum
%             indext=find(impfactor(i,:)==1);
%             indext=indext((indext>i));    
%             indext=unique(indext);
%             if(~isempty(indext))
%                 ed=ones(length(indext),2);
%                 ed(:,2)=i*ed(:,2);
%                 ed(:,1)=indext;
%                 edges8=[edges8;ed];
%             end
%         end
    %%
        impfactor = impfactor8;
        for i=1:length(bdIds)
            for j=i+1:length(bdIds)
            impfactor(bdIds(i),bdIds(j))=1;
            impfactor(bdIds(j),bdIds(i))=1;
            end
        end
        edges=[];
        for i=1:supNum
            indext=[];
            ind=find(impfactor(i,:)==1);
            for j=1:length(ind)
                indj=find(impfactor(ind(j),:)==1);
                indext=[indext,indj];
            end
            indext=[indext,ind];
            indext=indext((indext>i));    
            indext=unique(indext);
            if(~isempty(indext))
                ed=ones(length(indext),2);
                ed(:,2)=i*ed(:,2);
                ed(:,1)=indext;
                edges=[edges;ed];
            end
        end
        disp(['Run SCA']);
%     for layer1 = 2 : 32
%         disp([num2str(ii),'_',num2str(layer1)]);
        tic;
%         if layer == 37
%             continue;
%         end
%         process = [num2str(ii),'_',num2str(layer)];
%         disp(process);
%         saldir = ['.\vgg',num2str(layer),'pm_PASCALS\'];
%         if ~exist(saldir, 'dir')
%             mkdir(saldir);
%         end
        
%         layer = 37;
% 36 21 6 : 5 2 3
        matName = [matRoot,'\',imnames(ii).name(1:end-4),'.mat'];
        eval(['load ',matName,';']);
        layer1 = 32;
        layer2 = 6;
%         layer3 = 21;
        alpha1 = 5;
        alpha2 = 3;
        alpha3 = 2;
        alpha = [alpha1;alpha2;alpha3];
% 
%         matName = [matRoot,num2str(layer1),'\',imnames(ii).name(1:end-4),'.mat'];
%         eval(['load ',matName,';']);
%         vgg_feat1 = temp;
%         vgg_feat1 = double(imresize(vgg_feat1,[m n]));
%         meanVgg1 = GetMeanColor(vgg_feat1,pixelList,'vgg');
%           for layer1 = 33:42
          temp = layer32;
          vgg_feat1 = temp(padding(layer1):end-padding(layer1)+1,padding(layer1):end-padding(layer1)+1,:);
          vgg_feat1 = double(imresize(vgg_feat1,[m,n]));
          meanVgg1 = GetMeanColor(vgg_feat1,pixelList,'vgg');
          
          temp = layer6;
          vgg_feat2 = temp(padding(layer2):end-padding(layer2)+1,padding(layer2):end-padding(layer2)+1,:);
          vgg_feat2 = double(imresize(vgg_feat2,[m,n]));
          meanVgg2 = GetMeanColor(vgg_feat2,pixelList,'vgg');
%         matName = [matRoot,num2str(layer2),'\',imnames(ii).name(1:end-4),'.mat'];
%         eval(['load ',matName,';']);
%         vgg_feat2 = temp;
%         vgg_feat2 = double(imresize(vgg_feat2,[m,n]));
%         meanVgg2 = GetMeanColor(vgg_feat2,pixelList,'vgg');
%         
%         matName = [matRoot,num2str(layer3),'\',imnames(ii).name(1:end-4),'.mat'];
%         eval(['load ',matName,';']);
%         vgg_feat3 = temp;
%         vgg_feat3 = double(imresize(vgg_feat3,[m,n]));
%         meanVgg3 = GetMeanColor(vgg_feat3,pixelList,'vgg');
%         weights = makeweights(edges,meanVgg1,theta);
        weights = my_makeweights(edges,meanVgg1,meanLabCol,theta,alpha);
% %%
%         edges_weights = makeEdgeWeights(edges8,imedge,SupBound,theta2);
%         index = findSameEdgeInd(edges,edges8);
 %%       

%         weights = my_makeweights3(edges,meanVgg1,meanVgg2,meanVgg3,theta,alpha);
%         temp = zeros(length(edges),1);
%         temp(index) = edges_weights;
%         edges_weights = exp(3*temp);
%         edges_weights = 1;
%         weights = my_makeweights(edges,meanVgg1,meanVgg2,theta,alpha,edges_weights);
%         weights = weights.*temp;
        F = adjacency(edges,weights,supNum);                   

        % calculate a row-normalized impact factor matrix
        D_sam = sum(F,2);
        D = diag(D_sam);
        F_normal = D \ F;   % the row-normalized impact factor matrix

        % compute Coherence Matrix 
        C = a * normalization(1./max(F),0) + b;
        C_normal = diag(C);

        %%-----------------Single-layer Cellular Automata---------------%%
    %%
        %     S_prior=normalization(S_prior,0);
        S_N1=S_prior;
    %     S_N1 = C';
    %     S_N1(bg_seeds)=0;

    %     image_sam_1=zeros(m,n);
    %     image_sam_1(:)=S_N1(sulabel(:));
    %     image_saliency_1 = zeros(w(1), w(2));
    %     image_saliency_1(w(3):w(4), w(5):w(6)) = image_sam_1;
    %     figure;
    %     set (gcf,'Position',[1000,0,n,m]);
    %     imshow(image_saliency_1);
    %     outname=[saldir imnames(ii).name(1:end-4) '_AV_SCA_prior' '.png'];
    %     imwrite(image_saliency_1,outname);
        diff = setdiff(1:supNum, bdIds);

        % step1: decrease the saliency value of boundary superpixels
%         S_N1(bdIds) = S_N1(bdIds) - 0.6;
%         neg_Ind = find(S_N1 < 0);
%         if numel(neg_Ind) > 0
%            S_N1(neg_Ind) = 0.001; 
%         end
        for lap=1:5
            S_N1(bdIds) = S_N1(bdIds) - 0.6;
            neg_Ind = find(S_N1 < 0);
            if numel(neg_Ind) > 0
               S_N1(neg_Ind) = 0.001; 
            end
            S_N1=C_normal*S_N1+(1-C_normal).*diag(ones(1,supNum))*F_normal*S_N1;
%             S_N1 = F_normal*S_N1;
            S_N1(diff)=normalization(S_N1(diff),0);
%             image_sam_1=zeros(m,n);
%             image_sam_1(:)=S_N1(sulabel(:));
%             image_saliency_1 = zeros(w(1), w(2));
%             image_saliency_1(w(3):w(4), w(5):w(6)) = image_sam_1;
% %             figure;
% %             set (gcf,'Position',[1000,0,n,m]);
%             imshow(image_saliency_1);
        end        
    %     image_sam_1=zeros(m,n);
    %     image_sam_1(:)=S_N1(sulabel(:));
    %     image_saliency_1 = zeros(w(1), w(2));
    %     image_saliency_1(w(3):w(4), w(5):w(6)) = image_sam_1;
    % %     figure;
    % %     set (gcf,'Position',[0,500,n,m]);
    % %     imshow(image_saliency_1);
    %     outname=[saldir1 imnames(ii).name(1:end-4) '.png'];
    %     imwrite(image_saliency_1,outname);
        % step2: control the ratio of foreground larger than a threshold
        for lap = 1:5
            S_N1(bdIds) = S_N1(bdIds) - 0.6;
            neg_Ind = find(S_N1 < 0);
            if numel(neg_Ind) > 0
               S_N1(neg_Ind) = 0.001; 
            end
            most_sal_sup = find(S_N1 >0.93);
            if numel(most_sal_sup) < 0.02*supNum
                sal_diff = setdiff(1:supNum, most_sal_sup);
                S_N1(sal_diff) = normalization(S_N1(sal_diff),0);
            end
            S_N1=C_normal*S_N1+(1-C_normal).*diag(ones(1,supNum))*F_normal*S_N1;
%             S_N1 = F_normal*S_N1;

            S_N1(diff)=normalization(S_N1(diff),0);
%             image_sam_1=zeros(m,n);
%             image_sam_1(:)=S_N1(sulabel(:));
%             image_saliency_1 = zeros(w(1), w(2));
%             image_saliency_1(w(3):w(4), w(5):w(6)) = image_sam_1;
% %             figure;
% %             set (gcf,'Position',[1000,0,n,m]);
%             imshow(image_saliency_1);
        end  
    %     image_sam_1=zeros(m,n);
    %     image_sam_1(:)=S_N1(sulabel(:));
    %     image_saliency_1 = zeros(w(1), w(2));
    %     image_saliency_1(w(3):w(4), w(5):w(6)) = image_sam_1;
    % %     figure;
    % %     set (gcf,'Position',[1000,500,n,m]);
    % %     imshow(image_saliency_1);
    %     outname=[saldir2 imnames(ii).name(1:end-4) '.png'];
    %     imwrite(image_saliency_1,outname);
        % step3: simply update the saliency map according to rules
        for lap = 1:10
            S_N1 = C_normal*S_N1+(1-C_normal).*diag(ones(1,supNum))*F_normal*S_N1;
%             S_N1 = F_normal*S_N1;

            S_N1 = normalization(S_N1, 0);
%             image_sam_1=zeros(m,n);
%             image_sam_1(:)=S_N1(sulabel(:));
%             image_saliency_1 = zeros(w(1), w(2));
%             image_saliency_1(w(3):w(4), w(5):w(6)) = image_sam_1;
% %             figure;
% %             set (gcf,'Position',[1000,0,n,m]);
%             imshow(image_saliency_1);
        end

        image_sam=zeros(m,n);
        image_sam(:)=S_N1(sulabel(:));
        image_saliency = zeros(w(1), w(2));
        image_saliency(w(3):w(4), w(5):w(6)) = image_sam;
    %     figure;
    %     set (gcf,'Position',[0,0,n,m]);
    %     imshow(image_saliency);
        outpath = [saldir];
        outname=[outpath imnames(ii).name(1:end-4) '.png'];
        imwrite(image_saliency,outname);
%         imshow(image_saliency);
     %%
     toc;
     disp(['....']);
%     end
end