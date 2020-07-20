clear all;
global imgRoot;
global imgRoot1;
global imgRoot2;
global supdir;
global mu;
global theta1;
global theta2;
global u;
global gama;
mu=1e-3;
theta1=40;
theta2=40;
u=0.99;
gama=1.7;
imgRoot='E:\WORK1\MY CODE\主成分分析mutiple scale without noise\test\';
imgRoot1='E:\WORK1\MY CODE\主成分分析mutiple scale without noise\test\G\';%%可见光输入
imgRoot2='E:\WORK1\MY CODE\主成分分析mutiple scale without noise\test\T\';%%热红外输入
supdir='./superpixels/';
if(~exist(imgRoot,'dir'))
    mkdir(imgRoot);
end
if(~exist(supdir,'dir'))
   
    mkdir(supdir);
end
    cd 'E:\WORK1\MY CODE\主成分分析mutiple scale without noise';
     saldir1=['E:\WORK1\MY CODE\主成分分析mutiple scale without noise/first/'];%%输出
    saldir2=['E:\WORK1\MY CODE\主成分分析mutiple scale without noise/results/'];%%输出
    sal=dir([saldir2 '*' 'png']); 
    if(~exist(saldir2,'dir'))        
        mkdir(saldir1); 
        mkdir(saldir2); 
    end       
    if(length(sal)<821)
        DEMO11(saldir1, saldir2 )
    end

 
