clear all;
global imgRoot;
global imgRoot1;
global imgRoot2;
global supdir;
global saldir;
global theta1;
global theta2;
global spnumber;
global Gama1;
global Gama2;
global beta1;
global beta2;

spnumber=300;
u=0.02;
u1=0.06;
lambda=0.02;
theta1=24;
theta2=12;
Gama1=3;
Gama2=6;
beta1=0.15;
beta2=0.34;

A=cell(169,10);
temp=1;
imgRoot='./share/5000/';
imgRoot1='E:\data\VT5000\Test\RGB\';% test grayscale image path
imgRoot2='E:\data\VT5000\Test\T\';% test grayscale image path
supdir='./superpixels/300/';% the superpixel label file path
mkdir(imgRoot);
mkdir(supdir);

% for lambda=0.02:0.01:0.04
saldir='./saliencymap/';% the output path of the saliency map   
mkdir(saldir); 
sal=dir([saldir '*' 'png']);
DEMO11(lambda,u,u1);  
%cd 'C:\Users\Administrator\Desktop\TMM-CODE\RGB-T Saliency\multiM2'; %%¸ÄÂ·¾¶
%[Pre,Recall,PreF,RecallF,FMeasureF]=Plot_PreRecallThousand_contrast(u1,lambda,saldir);

% end
