%%
clear;
%存放方法图像
% Mapdir = 'D:\workspace\manifold Ranking progress train result\manifold ranking ASD1000';
% Mapdir = 'D:\workspace\manifold Ranking progress train result\manifold ranking SOD300';
%  Mapdir = 'D:\workspace\manifold Ranking progress train result\manifold ranking CSSD200';
Mapdir = 'E:\WORK1\code/theta1=40;theta2=40;gama=1.8;\';

  % Mapdir = 'C:\Documents and Settings\Administrator\桌面\Code\markov\2 Markov four edges\TCOFencymap\' ; %结果图目录 记得加最后的\
%存放真值图像
% GroundDir = 'D:\workspace\database_ground truth\ASD MSRA1000';
 %  GroundDir = 'D:\workspace\database_ground truth\SOD300';
   % GroundDir = 'F:\显著性相关Datasets\RGBT821datasets\GT\1';
 GroundDir = 'F:\显著性相关Datasets\RGB-Tdataset1000\GT';   

    cd(GroundDir);
  ImgEnum=dir('*.jpg');   
ImgNum=length(ImgEnum);%图像数量
  Pre = zeros(21,1);
  Recall = zeros(21,1);  
%   FMeasure = zeros(21,1);
  jj=0;    PreF = 0; RecallF = 0; FMeasureF = 0;   FigAnd = 0;  bigthreshold = 0; ThresholdAnd = 0;
for i=1:ImgNum     
    i   
    cd(GroundDir);
    Binary = imread( ImgEnum(i).name );
    NumOne= length( find(Binary(:,:,1) >0) );
    [height,width] = size( Binary(:,:,1) );  
    
    cd (Mapdir);
    mapImg = imread( strcat( ImgEnum(i).name(1:end-4),'_stage2.png' ) );    
%     Label2 = imread( ImgEnum(i).name );
    Label2 = mat2gray( mapImg );
   
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
 % cd 'C:\Documents and Settings\Administrator\桌面\Code\15 Markov\SubCode';
  Pre = Pre ./jj ;
  Recall = Recall ./jj;  
%   FMeasure = FMeasure ./ jj;
  
  PreF = PreF /jj 
  RecallF = RecallF /jj
  FMeasureF = FMeasureF / jj
    save( ['out.mat']  , 'Pre', 'Recall', 'FMeasureF','PreF','RecallF');       %  'FMeasure',

  figure(1);
    hold on;
  plot(Recall ,Pre,  'g' );  
  axis([0 1 0 1])
  xlabel('Recall');
  ylabel('Precision');  
  grid on; 
 hold off;
 
%  %figure(2);
%  hold on;
%  plot(Pre,  'r' );
%  xlabel('Threshold');
%  ylabel('Pre');
%  grid on;
%  hold off;

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
         
   