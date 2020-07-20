 function DEMO11( lambda , u, u1)
 %% 主函数
global spnumber;
global imgRoot;
global imgRoot1;
global imgRoot2;
global supdir;
global saldir;
global theta1;
global theta2;
global Gama1;
global Gama2;
global beta1;
global beta2;

imnames1=dir([imgRoot1 '*' 'jpg']); %%读RGB图像
imnames2=dir([imgRoot2 '*' 'jpg']); %%读thermal图像

%% 两种模态得到相同的SLIC过分割的 superpixel 图

 for ii=1:length(imnames1)   
    r1=0.5;%%模态1初始权重
    r2=0.5;%%模态2初始权重 
    im1=[imgRoot1 imnames1(ii).name];
    im2=[imgRoot2 imnames2(ii).name];     
    img1=imread(im1);
    img2=imread(im2);   
    Simg=0.5*img1 +0.5* img2;%%将要过分割的叠加后的图
    Simgn=[imgRoot imnames1(ii).name];
    Simgname=[Simgn(1:end-4)  '.bmp'];
    imwrite(Simg,Simgname,'bmp');
	[m,n,k]=size(Simg); 
    w=[m,n,1,m,1,n];
    %%----------------------generate superpixels--------------------%%   
    comm=['SLICSuperpixelSegmentation' ' ' Simgname ' ' int2str(20) ' ' int2str(spnumber) ' ' supdir];%%过分割叠加后的图，SLIC后的图存放在superpixels文件夹下 
	system(comm);
    spname=[supdir imnames1(ii).name(1:end-4)  '.dat'];%%在在superpixels文件夹下，生成.dat文件
    superpixels=ReadDAT( [m,n],spname);
    spnum=max(superpixels(:));% the actual superpixel number  

    % get edges(直接邻接和间接邻接边)
           adjMerge=AdjcProcloop(superpixels,spnum); %获得8连通图，是一个矩阵
           edges=[];%存放直接邻接的边
           for i=1:spnum
               indext=[];
               ind=find(adjMerge(i,:)==1);%直接邻接的
               for j=1:length(ind) %间接邻接的
                   indj=find(adjMerge(ind(j),:)==1);
                   indext=[indext,indj];
               end
               indext=[indext,ind];
               indext=indext((indext>i));
               indext=unique(indext);%去除自己和重复的
               if(~isempty(indext))
                   ed=ones(length(indext),2);
                   ed(:,2)=i*ed(:,2);   
                   ed(:,1)=indext;
                   edges=[edges;ed];                   
               end
           end 
           
   for kk=1:2
      %% -------------- 模态1---------------- %%
       if(kk==1)
           input_vals1=reshape(img1, m*n, k);
           rgb_vals1=zeros(spnum,1,3);
           inds=cell(spnum,1);
           for i=1:spnum
               inds{i}=find(superpixels==i);
               rgb_vals1(i,1,:)=mean(input_vals1(inds{i},:),1); 
           end
           lab_vals1 = colorspace('Lab<-', rgb_vals1);
           seg_vals1=reshape(lab_vals1,spnum,3);% lab 颜色特征
         
           
          Weights1 = makeweights(edges,seg_vals1,theta1); 
          W1 = adjacency(edges,Weights1,spnum);  
          dd1 = sum(W1); D1 = sparse(1:spnum,1:spnum,dd1); clear dd1;  
       end%%结束第一个模态
 
%% ------------ 模态2---------------- %%
       if(kk==2)   
           input_vals2=reshape(img2, m*n, k);
           rgb_vals2=zeros(spnum,1,3);    
           for i=1:spnum  
               rgb_vals2(i,1,:)=mean(input_vals2(inds{i},:),1);  
           end
           lab_vals2 = colorspace('Lab<-', rgb_vals2);
           seg_vals2=reshape(lab_vals2,spnum,3);% feature for each superpixel           
          
          Weights2 = makeweights(edges,seg_vals2,theta2); 
          W2= adjacency(edges,Weights2,spnum);                   
          dd2 = sum(W2); D2 = sparse(1:spnum,1:spnum,dd2); clear dd2; 
         
%% -----------求相关矩阵-------------------------%%
       DD1=full(D1)^(-1/2);
       DD2=full(D2)^(-1/2);
       L1=DD1*full(W1)*DD1;       
       L2=DD2*full(W2)*DD2;
       A1=eye(spnum)-L1;
       A2=eye(spnum)-L2;
       C=[eye(spnum, spnum),-eye(spnum, spnum)];       

   %% ----------------stage1--------------%% 
         Init_r1=r1;
         Init_r2=r2;
         temp1=[];
         temp2=[];    
         
         %% top
         Yt=zeros(spnum,1);
         bst=unique(superpixels(1,1:n));         
         Yt(bst)=1;
         YYt=[Yt;Yt];   %%top seeds2？？？？？？
         [bsalt, r1, r2] = CMR( YYt,  Init_r1, Init_r2, A1, A2,  C, u, lambda, spnum);%%迭代更新top的ranking
         temp1=[temp1;r1];
         temp2=[temp2;r2];
         bsalt=(bsalt-min(bsalt(:)))/(max(bsalt(:))-min(bsalt(:)));
         bsalt=1-bsalt;  
         
          %% down
         Yd=zeros(spnum,1);      
         bsd=unique(superpixels(m,1:n));
         Yd(bsd)=1;
         YYd=[Yd;Yd];  
         [bsald, r1, r2] = CMR( YYd, Init_r1, Init_r2, A1, A2,  C, u, lambda, spnum);%%迭代更新down的ranking
         temp1=[temp1;r1]; 
         temp2=[temp2;r2];
         bsald=(bsald-min(bsald(:)))/(max(bsald(:))-min(bsald(:)));
         bsald=1-bsald; 
         
           %% right
         Yr=zeros(spnum,1);
         bsr=unique(superpixels(1:m,1));
         Yr(bsr)=1;
         YYr=[Yr;Yr]; 
         [bsalr, r1, r2] = CMR( YYr,  Init_r1, Init_r2, A1, A2,  C, u, lambda, spnum);%%迭代更新right的ranking
         temp1=[temp1;r1]; 
         temp2=[temp2;r2];
         bsalr=(bsalr-min(bsalr(:)))/(max(bsalr(:))-min(bsalr(:)));
         bsalr=1-bsalr;
         %% left
         Yl=zeros(spnum,1);
         bsl=unique(superpixels(1:m,n));
         Yl(bsl)=1;
         YYl=[Yl;Yl]; 
         [bsall, r1, r2] = CMR( YYl, Init_r1, Init_r2, A1, A2,  C, u, lambda, spnum);%%迭代更新left的ranking
         temp1=[temp1;r1]; 
         temp2=[temp2;r2];
         bsall=(bsall-min(bsall(:)))/(max(bsall(:))-min(bsall(:)));
         bsall=1-bsall; 
         r1=mean(temp1);
         r2=mean(temp2);
          %% combine 
         bsalc=(bsalt.*bsald.*bsall.*bsalr);
         bsalc=(bsalc-min(bsalc(:)))/(max(bsalc(:))-min(bsalc(:)));
          %% 取出bsalc1,bsalc2
         bsalc1=bsalc(1:spnum);
         bsalc2=bsalc(spnum+1:2*spnum); 
          %% show the result of  stage1          
         bsalc_stage1=r1*bsalc1+r2*bsalc2; 
        tmapstage1=zeros(m,n);
        tmapstage1_1=zeros(m,n);
        tmapstage1_2=zeros(m,n);
        for i=1:spnum
            tmapstage1(inds{i})=bsalc_stage1(i);
            tmapstage1_1(inds{i})=bsalc1(i);
            tmapstage1_2(inds{i})=bsalc2(i);
        end
         tmapstage1=(tmapstage1-min(tmapstage1(:)))/(max(tmapstage1(:))-min(tmapstage1(:)));
         mapstage1=zeros(w(1),w(2));
         mapstage1(w(3):w(4),w(5):w(6))=tmapstage1;
         mapstage1=uint8(mapstage1*255);  
%          outname=[saldir imnames1(ii).name(1:end-4) '_stage1.png'];
%          imwrite( mapstage1 , outname);

         
         
         %% ---------------stage2---------------%%
         seeds1=bsalc1;
         seeds2=bsalc_stage1;
         threshold1 = max(seeds1)-beta1;
         threshold2 = max(seeds2)-beta2;
         seeds1(seeds1<threshold1)=0;
         seeds1(seeds1>threshold1)=1;
         seeds2(seeds2<threshold2)=0;
         seeds2(seeds2>threshold2)=1;%% obtain foreground seeds2 
 
         
         r1=0.5;
         r2=0.5;
         seeds=[seeds1;seeds2];

         %% 第二阶段迭代更新
         [fsal,r1,r2] = CMR( seeds , r1, r2, A1, A2,  C, u1, lambda, spnum);
         fsal1=fsal(1:spnum);
         fsal2=fsal(spnum+1 : 2*spnum);  
         
%          fsal1=(fsal1-min(fsal1(:)))/(max(fsal1(:))-min(fsal1(:)));
%          fsal2=(fsal2-min(fsal2(:)))/(max(fsal2(:))-min(fsal2(:)));

        bsalc_stage2=r1*fsal1+r2*fsal2;
        tmapstage2=zeros(m,n);
        tmapstage2_1=zeros(m,n);
        tmapstage2_2=zeros(m,n);
        for i=1:spnum
            tmapstage2(inds{i})=bsalc_stage2(i);
            tmapstage2_1(inds{i})=fsal1(i);
            tmapstage2_2(inds{i})=fsal2(i);
        end
        tmapstage2=(tmapstage2-min(tmapstage2(:)))/(max(tmapstage2(:))-min(tmapstage2(:)));
        mapstage2=zeros(w(1),w(2));
        mapstage2(w(3):w(4),w(5):w(6))=tmapstage2;
        mapstage2=uint8(mapstage2*255);  
        outname=[saldir imnames2(ii).name(1:end-4) '_stage2.png'];
        imwrite(mapstage2 , outname);   
        
%         tmapstage2_1=(tmapstage2_1-min(tmapstage2_1(:)))/(max(tmapstage2_1(:))-min(tmapstage2_1(:)));
%         mapstage2_1=zeros(w(1),w(2));
%         mapstage2_1(w(3):w(4),w(5):w(6))=tmapstage2_1;
%         mapstage2_1=uint8(mapstage2_1*255);  
%         outname_1=[saldir imnames2(ii).name(1:end-4) '_stage2(模态1).png'];
%         imwrite(mapstage2_1 , outname_1); 
%         
%         tmapstage2_2=(tmapstage2_2-min(tmapstage2_2(:)))/(max(tmapstage2_2(:))-min(tmapstage2_2(:)));
%         mapstage2_2=zeros(w(1),w(2));
%         mapstage2_2(w(3):w(4),w(5):w(6))=tmapstage2_2;
%         mapstage2_2=uint8(mapstage2_2*255);  
%         outname_2=[saldir imnames2(ii).name(1:end-4) '_stage2(模态2).png'];
%         imwrite(mapstage2_2 , outname_2); 
             
        
       end%%结束第二个模态
   end

 end%%第ii张ranking结束

 
 end %%结束整个函数












          
          
          
          
          
          
          
          