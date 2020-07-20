clear all;
global imgRoot;
global imgRoot1;
global imgRoot2;
global supdir;
global saldir;
global theta1;
global theta2;
global spnumber;
global Gama;
global beta;

spnumber=300;
u=0.02;
u1=0.06;
lambda=0.03;
theta1=24;
theta2=12;
Gama=5;
beta=0.25;

A=cell(169,10);
temp=1;
imgRoot='./share/1/';
imgRoot1='./test/1/G/';% test grayscale image path
imgRoot2='./test/1/T/';% test grayscale image path
supdir='./superpixels/1/';% the superpixel label file path
mkdir(imgRoot);
mkdir(supdir);

% for u1=0.4:0.4
saldir=['./saliencymap/1/'];% the output path of the saliency map   
mkdir(saldir); 
sal=dir([saldir '*' 'png']);
if(length(sal)<1642)
    DEMO11(lambda,u,u1);  
end 
% cd 'C:/Users/wgz/Desktop/multiM2/';
% [Pre,Recall,PreF,RecallF,FMeasureF]=Plot_PreRecallThousand_contrast(u1,lambda,saldir);

% end
