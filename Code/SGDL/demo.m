 %%% Title：RGB-T Image Saliency Detection via Collaborative Graph Learning
%%% Author：Zhengzheng Tu, Tian Xia, Chenglong Li, Xiaoxiao Wang, Yan Ma and Jin Tang
%%Paper link: https://ieeexplore.ieee.org/document/8744296



%% 主函数
 clear;
tic;
imgRoot = './test/';
imgRoot1='E:/VT5000/VT5000/Test/RGB/';%'./test/RGB/';%%RGB images input
imgRoot2='E:/VT5000/VT5000/Test/T/';%'./test/T/';%%Thermal images input
saldir='./saliencymap/';% the output path of the saliency map
supdir='./superpixels/';% the superpixel label file path
FCNfeatureRoot1 = './FCN-feature/RGB/';%'./FCN-feature/RGB/';%%use pretrained FCN-32S network
FCNfeatureRoot2 = './FCN-feature/T/'; %'./FCN-feature/T/';
mkdir(supdir);
mkdir(saldir);
imnames1=dir([imgRoot1 '*' 'jpg']);
imnames2=dir([imgRoot2 '*' 'jpg']);
theta1=20;
theta2=40;
theta3=20;
theta4=40;
spnumber=300;
eta=1.8;
%% 两种模态得到相同的SLIC过分割的 superpixel 图
 for ii=1:length(imnames1)  %%这里假设假设两种模态图片数量是一样的 
     ii   
    im1=[imgRoot1 imnames1(ii).name];
    im2=[imgRoot2 imnames2(ii).name];     
    img1=imread(im1);
    img2=imread(im2);   
    Simg=0.5*img1 +0.5* img2; %%将要过分割的叠加后的图
    Simgn=[imgRoot imnames1(ii).name];
    Simgname=[Simgn(1:end-4)  '.bmp'];
    imwrite(Simg,Simgname,'bmp');% the slic software support only the '.bmp' image
       [m,n,k]=size(Simg);
   
 %%  ----------------------- SLICSuperpixelSegmentation ------------------------%%
           Simgname=[Simgn(1:end-4)  '.bmp'];
           comm=['SLICSuperpixelSegmentation' ' ' Simgname ' ' int2str(20) ' ' int2str(spnumber) ' ' supdir];%%过分割叠加后的图，SLIC后的图存放在superpixels文件夹下
            system(comm);
           spname=[supdir imnames1(ii).name(1:end-4)  '.dat'];%%在在superpixels文件夹下，生成.dat文件
           superpixels=ReadDAT( [m,n],spname);
           spnum=max(superpixels(:));
 %%  ----------------------- get edges-------------------------%%                
 adjloop=AdjcProcloop(superpixels,spnum);
    edges=[];
    for i=1:spnum
        indext=[];
        ind=find(adjloop(i,:)==1);
        for j=1:length(ind)
            indj=find(adjloop(ind(j),:)==1);
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
  inds = cell(spnum,1);
    for ttt=1:spnum
        inds{ttt} = find(superpixels==ttt);
    end
   
    %%  ----------------------compute RGB modality affinity matrix-------------------------%% 
   [G_meanVgg1,G_meanVgg2] = ExtractFCNfeature(FCNfeatureRoot1,imnames1(ii).name(1:end-4),inds,m,n);       
  
    G_weights1 = makeweights(edges,G_meanVgg1,theta1);    
    G_weights2 = makeweights(edges,G_meanVgg2,theta1);   
    G_W1 = adjacency(edges,G_weights1,spnum); 
    G_W2 = adjacency(edges,G_weights2,spnum);
    dd = sum(G_W1); G_D1 = sparse(1:spnum,1:spnum,dd); clear dd;
    G_L1 =G_D1-G_W1; 
    dd = sum(G_W2); G_D2 = sparse(1:spnum,1:spnum,dd); clear dd;
    G_L2 =G_D2-G_W2; 
     %%  ----------------------compute thermal modality affinity matrix-------------------------%%  
   [T_meanVgg1,T_meanVgg2] =ExtractFCNfeature(FCNfeatureRoot2,imnames2(ii).name(1:end-4),inds,m,n);
    T_weights1 = makeweights(edges,T_meanVgg1,theta2);    
    T_weights2 = makeweights(edges,T_meanVgg2,theta2);   
    T_W1 = adjacency(edges,T_weights1,spnum); 
    T_W2 = adjacency(edges,T_weights2,spnum);
    dd = sum(T_W1); T_D1 = sparse(1:spnum,1:spnum,dd); clear dd;
    T_L1 =T_D1-T_W1; 
    dd = sum(T_W2); T_D2 = sparse(1:spnum,1:spnum,dd); clear dd;
   T_L2 =T_D2-T_W2; 
   
    %% traditional feature
   input_vals1 = reshape(img1, m*n, k); 
   input_vals2 = reshape(img2, m*n, k);
   rgb_vals1 = zeros(spnum,1,3);
   rgb_vals2 = zeros(spnum,1,3); 
   for i = 1:spnum
        rgb_vals1(i,1,:) = mean(input_vals1(inds{i},:),1);  
        rgb_vals2(i,1,:) = mean(input_vals2(inds{i},:),1);
   end
   lab_vals1 = colorspace('Lab<-', rgb_vals1);
   lab_vals2 = colorspace('Lab<-', rgb_vals2);
   seg_vals1=reshape(lab_vals1,spnum,3);% lab 颜色特征   
   seg_vals2=reshape(lab_vals2,spnum,3);
    Weights1 = makeweights(edges,seg_vals1,theta3); 
   Weights2 = makeweights(edges,seg_vals2,theta4); 
   W1 = adjacency(edges,Weights1,spnum);  
   W2= adjacency(edges,Weights2,spnum);     
   dd1 = sum(W1); D1 = sparse(1:spnum,1:spnum,dd1);clear dd1;
   dd2 = sum(W2); D2 = sparse(1:spnum,1:spnum,dd2); clear dd2;
   G_L3=D1-W1;
   T_L3=D2-W2;

        %% ------------------------stage1----------------------%%      
     %% top
   
      Yt=zeros(spnum,1);
      bst=unique(superpixels(1,1:n));         
      Yt(bst)=1;
    [St] =Learn(G_L1,G_L2,G_L3,T_L1,T_L2,T_L3,spnum,Yt);   
      St=(St-min(St(:)))/(max(St(:))-min(St(:)));
      St=1-St;  

    %% down
     Yd=zeros(spnum,1);
     bst=unique(superpixels(m,1:n));         
     Yd(bst)=1;
   [Sd] =Learn(G_L1,G_L2,G_L3,T_L1,T_L2,T_L3,spnum,Yd);    
       Sd=(Sd-min(Sd(:)))/(max(Sd(:))-min(Sd(:)));
     Sd=1-Sd;
    %% right
     Yr=zeros(spnum,1);
     bst=unique(superpixels(1:m,1));         
     Yr(bst)=1;
   [Sr] =Learn(G_L1,G_L2,G_L3,T_L1,T_L2,T_L3,spnum,Yr);    
       Sr=(Sr-min(Sr(:)))/(max(Sr(:))-min(Sr(:)));
     Sr=1-Sr; 
    %% left
     Yl=zeros(spnum,1);
     bst=unique(superpixels(1:m,n));         
     Yl(bst)=1;
 [Sl] =Learn(G_L1,G_L2,G_L3,T_L1,T_L2,T_L3,spnum,Yl); 
        Sl=(Sl-min(Sl(:)))/(max(Sl(:))-min(Sl(:)));
     Sl=1-Sl;
   %% combine 
    Sc=(St.*Sd.*Sl.*Sr);
    Sc=(Sc-min(Sc(:)))/(max(Sc(:))-min(Sc(:))); 
%   %% show the result of  stage1          
%     mapstage1=zeros(m,n);
%     for i=1:spnum
%         mapstage1(inds{i})=Sc(i);
%     end
%     mapstage1=(mapstage1-min(mapstage1(:)))/(max(mapstage1(:))-min(mapstage1(:)));
%     mapstage1=uint8(mapstage1*255);  
%     outname=[saldir imnames1(ii).name(1:end-4) '_stage1.png'];
%     imwrite( mapstage1 , outname); 
   %% seeds的选取
     seeds=Sc;
     th=mean(Sc)*eta;
     seeds(seeds<th)=0;
     seeds(seeds>=th)=1;
    %% ------------------------stage2----------------------%%   
   [S] =Learn(G_L1,G_L2,G_L3,T_L1,T_L2,T_L3,spnum,seeds);  %迭代更新top的ranking
    mapstage2=zeros(m,n);
    for i=1:spnum
        mapstage2(inds{i})=S(i);
    end
    mapstage2=(mapstage2-min(mapstage2(:)))/(max(mapstage2(:))-min(mapstage2(:)));
    mapstage2=uint8(mapstage2*255);  
    outname=[saldir imnames1(ii).name(1:end-4) '_stage2.png'];
    imwrite(mapstage2 , outname); 
toc;
 end%%结束整个函数

 