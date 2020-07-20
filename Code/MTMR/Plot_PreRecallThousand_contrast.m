function [Pre,Recall,PreF,RecallF,FMeasureF]=Plot_PreRecallThousand_contrast(u1,lambda,Mapdir)
%%
global spnumber;
global theta1;
global theta2;
global spnumber;
global Gama;
global beta;

Mapdir ='E:\copyy\实验数据\对比实验总结果（新）\对比方法总结果\fuse(10类)\CA ';
GroundDir = 'E:\RGB-T Saliency\multiM2\test\821\GT';

%  GroundDir = 'C:\Users\wgz\Desktop\multiM\test\dataset(new)\GT';

 %  GroundDir = 'D:\workspace\database_ground truth\SOD300';
%  GroundDir = 'D:\workspace\database_ground truth\CSSD200';
  % GroundDir = 'D:\workspace\database_ground truth\ECSSD999';   
  cd (GroundDir); 
ImgEnum=dir('*.jpg');   ImgNum=length(ImgEnum);%图锟斤拷锟斤拷锟斤拷
  Pre = zeros(21,1);
  Recall = zeros(21,1);  
%   FMeasure = zeros(21,1);
  jj=0;    PreF = 0; RecallF = 0; FMeasureF = 0;   FigAnd = 0;  bigthreshold = 0; ThresholdAnd = 0;
  
for i=1:ImgNum     
    cd(GroundDir);
    Binary = imread( ImgEnum(i).name );
    NumOne= length( find(Binary(:,:,1) >0) );
    [height,width] = size( Binary(:,:,1) );  
    
    cd (Mapdir);
    mapImg = imread( strcat( ImgEnum(i).name(1:end-4),'_CA.png' ) );    
%     Label2 = imread( ImgEnum(i).name );
    Label1 = mapImg;
    Label2 = mat2gray(Label1  );
   
%%  thou berke Pre recall      
  if NumOne ~= 0
      jj=jj+1;   mm = 1;     
   for j = 0 : .05 : 1
       Label3 = zeros( height, width );  
       Label3( Label2>=j )=1;        
       NumRec = length( find( Label3==1 ) );
                
       LabelAnd = Label3 & Binary(:,:,1);   
       NumAnd = length( find( LabelAnd==1 ) );
       if NumAnd == 0
           FigAnd = FigAnd + 1;
           break;  
       end
       Pretem = NumAnd/NumRec;    
       Recalltem =  NumAnd/NumOne;
       
       Pre(mm) = Pre(mm) +  Pretem;  
       Recall(mm) = Recall(mm) + Recalltem;
       
%        FMeasure(mm) = FMeasure(mm) + ( (1 + .3) * Pretem * Recalltem ) / ( .3 * Pretem + Recalltem );
       mm = mm + 1;
   end
   
      sumLabel =  2* sum( sum(Label2) ) / (height*width) ;
%       sumLabel =  5* sum( sum(Label2) ) / (height*width) ;
      if ( sumLabel >= 1 )           
              sumLabel = .902 ;    bigthreshold = bigthreshold +1;
      end       
       
       Label3 = zeros( height, width );
       Label3( Label2>=sumLabel ) = 1;       

       NumRec = length( find( Label3==1 ) );
       
       LabelAnd = Label3 & Binary(:,:,1);        
       NumAnd = length( find ( LabelAnd==1 ) );
       
       if NumAnd == 0
           ThresholdAnd = ThresholdAnd +1;
           continue;
       end
       
       PreFtem = NumAnd/NumRec; 
       RecallFtem = NumAnd/NumOne;   
       
       PreF = PreF +    PreFtem;   
       RecallF = RecallF +    RecallFtem;
       
       FMeasureF = FMeasureF + ( ( ( 1 + .3) * PreFtem * RecallFtem ) / ( .3 * PreFtem + RecallFtem ) );
  end   

        
end
  %% Mean Pre Recall

FigAnd
ThresholdAnd
bigthreshold
%   cd 'E:\实锟斤拷锟斤拷锟絓cvpr13_Saliency Detection via Graph-Based Manifold Ranking--code\cvprcode\mapcodemap';
  Pre = Pre ./jj ;
  Recall = Recall ./jj;  
%   FMeasure = FMeasure ./ jj;
  
  PreF = PreF /jj 
  RecallF = RecallF /jj
  FMeasureF = FMeasureF / jj
  
   % save(['C:\Users\wgz\Desktop\CA\Entire.mat'], 'Pre', 'Recall', 'FMeasureF','PreF','RecallF');       %  'FMeasure',
  figure(1);
  hold on;
  plot(Recall ,Pre,  'b' );  
  axis([0 1 0 1])
  xlabel('Recall');
  ylabel('Precision');  
  grid on; 
  hold off;
  
% figure(2);
% hold on;
% plot(Pre,  'r' );
% xlabel('Threshold');
% ylabel('Pre');
% grid on;
% hold off;
% 
% figure(3);
% hold on;
% plot(Recall,  'r' );
% xlabel('Threshold');
% ylabel('Recall');
% grid on;
% hold off;
% 
% figure(4);
% hold on;
% plot( FMeasure,  'r' );
% xlabel('Threshold');
% ylabel('FMeasure');
% grid on;
% hold off;
end
         
   