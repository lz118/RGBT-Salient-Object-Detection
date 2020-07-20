 %% 主函数
 tic
 clear;
imgRoot1='F:\显著性相关Datasets\RGB-Tdataset1000\RGB/';%%可见光输入
imgRoot2='F:\显著性相关Datasets\RGB-Tdataset1000\T/';%%热红外输入
saldir='./theta1=40;theta2=40;gama=1.8;\';% the output path of the saliency map
supdir='./superpixels/';% the superpixel label file path
mkdir(supdir);
mkdir(saldir);
mu=1e-3; 
theta1=40;
theta2=40;
gama=1.8;
imnames1=dir([imgRoot1 '*' 'jpg']);
imnames2=dir([imgRoot2 '*' 'jpg']);


%% 两种模态得到相同的SLIC过分割的 superpixel 图
 for ii=1:length(imnames1)  %%这里假设假设两种模态图片数量是一样的 
     ii   
    im1=[imgRoot1 imnames1(ii).name];
    im2=[imgRoot2 imnames2(ii).name];     
    img1=imread(im1);
    img2=imread(im2);   
    Simg=0.5*img1 +0.5* img2;%%将要过分割的叠加后的图
    Simgn=[imgRoot1 imnames1(ii).name];
    Simgname=[Simgn(1:end-4)  '.bmp'];
    imwrite(Simg,Simgname,'bmp');% the slic software support only the '.bmp' image
       [m,n,k]=size(Simg);
    mapstage=zeros(m,n); 
    testmapstage=zeros(m,n);

        
  
    %% ----------------------generate superpixels--------------------%%   
    for v=1:3
%     spnumber=1/v*400;
spnumber=v*100;
    Simgname=[Simgn(1:end-4)  '.bmp'];
    comm=['SLICSuperpixelSegmentation' ' ' Simgname ' ' int2str(20) ' ' int2str(spnumber) ' ' supdir];%%过分割叠加后的图，SLIC后的图存放在superpixels文件夹下
    system(comm);
    spname=[supdir imnames1(ii).name(1:end-4)  '.dat'];%%在在superpixels文件夹下，生成.dat文件
    superpixels=ReadDAT( [m,n],spname);
    spnum=max(superpixels(:));% the actual superpixel number  
   
   %% -------------- 模态1，2提特征求W1，W2，D1，D2，B---------------- %%
   
   input_vals1 = reshape(img1, m*n, k); 
   input_vals2 = reshape(img2, m*n, k);
   rgb_vals1 = zeros(spnum,1,3);
   rgb_vals2 = zeros(spnum,1,3);
   inds = cell(spnum,1);
    for ttt=1:spnum
        inds{ttt} = find(superpixels==ttt);
   end
   for i = 1:spnum
        rgb_vals1(i,1,:) = mean(input_vals1(inds{i},:),1);  
        rgb_vals2(i,1,:) = mean(input_vals2(inds{i},:),1);
   end
   lab_vals1 = colorspace('Lab<-', rgb_vals1);
   lab_vals2 = colorspace('Lab<-', rgb_vals2);
   seg_vals1=reshape(lab_vals1,spnum,3);% lab 颜色特征   
   seg_vals2=reshape(lab_vals2,spnum,3);
    % get edges
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
   Weights1 = makeweights(edges,seg_vals1,theta1); 
   Weights2 = makeweights(edges,seg_vals2,theta2); 
   W1 = adjacency(edges,Weights1,spnum);  
   W2= adjacency(edges,Weights2,spnum);     
   dd1 = sum(W1); D1 = sparse(1:spnum,1:spnum,dd1);clear dd1;
   dd2 = sum(W2); D2 = sparse(1:spnum,1:spnum,dd2); clear dd2;
   %% ---------------求相关矩阵------------------%%   
    DD1{v}= full(D1)^(-1/2);
   DD2{v} = full(D2)^(-1/2);
   I{v}=eye(spnum,spnum);
   L1{v} =I{v}-DD1{v}*full(W1)*DD1{v};
   L2{v} = I{v}-DD2{v}*full(W2)*DD2{v};
    H1{v}=D1-W1;
    H2{v}=D2-W2;  
     index{v}=spnum;
   super{v}=superpixels;
  pixlist{v}=inds;
    end   
   [c]=weightc(index,super,pixlist); 
   %% ----------------stage1--------------%% 
             
     %% top
     for v=1:3
      Yt{v}=zeros(index{v},1);%yt1代表第一个模态的种子节点
       bst=unique(super{v}(1,1:n));         
      Yt{v}(bst)=1;
      
     %% down    
   Yd{v}=zeros(index{v},1);%yt1代表第一个模态的种子节点
     bst=unique(super{v}(m,1:n));         
     Yd{v}(bst)=1;
         %% right
        Yr{v}=zeros(index{v},1);%yt1代表第一个模态的种子节点
     bst=unique(super{v}(1:m,1));         
     Yr{v}(bst)=1;
       %% left
   Yl{v}=zeros(index{v},1);%yt1代表第一个模态的种子节点
     bst=unique(super{v}(1:m,n));         
     Yl{v}(bst)=1;
     end
     tic
    [st,St]=twoCMR(Yt,index,H1,H2,I,L1,L2,c);%迭代更新top的ranking
    [sd,Sd]=twoCMR( Yd,index,H1,H2,I,L1,L2,c);%%迭代更新down的ranking
    [sr,Sr]=twoCMR( Yr,index,H1,H2,I,L1,L2,c);%%迭代更新right的ranking
    [sl,Sl]=twoCMR( Yl,index,H1,H2,I,L1,L2,c);%%迭代更新left的rankinng

      %% combine 
      %各个scale的s，s1.s2.s3
     
 for v=1:3
 St{v}=(St{v}-min(St{v}(:)))/(max(St{v}(:))-min(St{v}(:)));
   St{v}=1-St{v};  
    Sd{v}=(Sd{v}-min(Sd{v}(:)))/(max(Sd{v}(:))-min(Sd{v}(:)));
        Sd{v}=1-Sd{v}; 
         Sr{v}=(Sr{v}-min(Sr{v}(:)))/(max(Sr{v}(:))-min(Sr{v}(:)));
        Sr{v}=1-Sr{v};  
        Sl{v}=(Sl{v}-min(Sl{v}(:)))/(max(Sl{v}(:))-min(Sl{v}(:)));
        Sl{v}=1-Sl{v};
    Sc{v}=(St{v}.*Sd{v}.*Sl{v}.*Sr{v});
    Sc{v}=(Sc{v}-min(Sc{v}(:)))/(max(Sc{v}(:))-min(Sc{v}(:))); %各个尺度的
 end
       st=(st-min(st(:)))/(max(st(:))-min(st(:)));
      st=1-st;  
       sd=(sd-min(sd(:)))/(max(sd(:))-min(sd(:)));
      sd=1-sd;  
       sr=(sr-min(sr(:)))/(max(sr(:))-min(sr(:)));
      sr=1-sr;  
       sl=(sl-min(sl(:)))/(max(sl(:))-min(sl(:)));
      sl=1-sl;  
 s=(st.*sd.*sl.*sr);
   s=(s-min(s(:)))/(max(s(:))-min(s(:))); %s 
   s1=s(1:index{2});
   s2=s(index{2}+1:2*index{2});
   ss=1/2*(s1+s2);
     %% show the result of  stage1          
     for tt=1:index{2}
         testmapstage(pixlist{2}{tt})=ss(tt);
     end
     testmapstage=(testmapstage-min(testmapstage(:)))/(max(testmapstage(:))-min(testmapstage(:)));
      testmapstage=uint8(testmapstage*255);
     outname=[saldir imnames1(ii).name(1:end-4) '_stage1.png'];
     imwrite(testmapstage, outname);
  
      %% seeds的选取
     for v=1:3
    seeds=Sc;
    th{v}=mean(Sc{v})*gama; 
     end
%  th{v}=max(Sc{v})*0.56;
    seeds{1}(seeds{1}<th{1})=0;
    seeds{1}(seeds{1}>=th{1})=1;
   seeds{2}(seeds{2}<th{2})=0;
    seeds{2}(seeds{2}>=th{2})=1;
     seeds{3}(seeds{3}<th{3})=0;
    seeds{3}(seeds{3}>=th{3})=1;

      %% 第二阶段的更新
 [fsal,FSAL]=lastCMR(seeds,index,H1,H2,I,L1,L2,c,s);%迭代更新top的ranking
  fsal1=fsal(1:index{2});
   fsal2=fsal(index{2}+1:2*index{2});
   fs=1/2*(fsal1+fsal2);
     for tt=1:index{2}
      mapstage(pixlist{2}{tt})=fs(tt);
     end
 mapstage=(mapstage-min(mapstage(:)))/(max(mapstage(:))-min(mapstage(:)));
  mapstage=uint8(mapstage*255); 
  outname=[saldir imnames1(ii).name(1:end-4) '_stage2.png'];
  imwrite(mapstage, outname);
     end
    toc;
 %%结束整个函数
 
 













          
          
          
          
          
          
          
          