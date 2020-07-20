function [s,S]=twoCMR(Y,index,H1,H2,I,L1,L2,c)
%% Description: iterative updating---stage1
global mu;
global max_mu;
global lambda1;
global lambda2;
global lambda;
global lambda3;
global lambda4;
lambda=0.01;%%%
lambda1=0.7;
lambda2=0.01;
lambda3=0.5;
lambda4=0.5;
mu=1e-3;
max_mu=1e10;
s=zeros(2*index{2},1);
temp_s=s;   
iter=1; 
maxIter=100;
for v=1:3
    LL{v}=blkdiag(L1{v},L2{v});%%正则化的拉普拉斯
    II{v}=blkdiag(I{v},I{v});
    cc{v}=blkdiag(c{v},c{v});
    X{v}=[I{v},-I{v}];
    
    y1{v}=zeros(index{v},1);
    y2{v}=zeros(index{v},1);
    
    S{v}=zeros(2*index{v},1);
    temp_S{v}=S{v};
    S1{v}=zeros(index{v},1);
    S2{v}=zeros(index{v},1);
    
    temp_f1{v}= zeros(index{v},1);
    temp_f2{v}= zeros(index{v},1);  %求f的结果，单独两个模态做
  
    q1{v}=zeros(index{v},1);
    q2{v}=zeros(index{v},1);
    temp_Q{v}=[Y{v};Y{v}]; %初始的Q等于初始的种子节点
    %这用于更新种子节点就是q尖，用前面的Y代表原始的q
        ERROR = [];
end
   while iter<=maxIter  %%迭代优化  
    %--------------------------- 更新f--------------------------    
    for v=1:3
        b1=(4*lambda1*H1{v}+mu*I{v})\(eye(index{v}));
        b2=(4*lambda1*H2{v}+mu*I{v})\(eye(index{v}));
        f1{v}=b1*(mu*q1{v}+y1{v});
        f2{v}=b2*(mu*q2{v}+y2{v});
        e1=f1{v}-y1{v}/mu;
        e2=f2{v}-y2{v}/mu;
        q1{v}= softthreshold(S1{v},e1,Y{v},lambda,mu/2,lambda2,index{v});
        q2{v}= softthreshold(S2{v},e2,Y{v},lambda,mu/2,lambda2,index{v});
        Q{v}=[q1{v};q2{v}];
        S{v}=(LL{v}/lambda+II{v}+lambda3/lambda*cc{v}'*cc{v}+lambda4/lambda*X{v}'*X{v})\(eye(2*index{v}))*(Q{v}+lambda3/lambda*cc{v}'*s);
         % S{v}=(LL{v}+lambda*II{v}+lambda3*cc{v}'*cc{v}+lambda4*X{v}'*X{v})\(eye(2*index{v}))*(lambda*Q{v}+cc{v}'*s);
        S{v}(S{v}<0)=0;
    S1{v}=S{v}(1:index{v});
    S2{v}=S{v}(index{v}+1:2*index{v});
    end
    s=1/3*(cc{1}*S{1}+cc{2}*S{2}+cc{3}*S{3});
  
       %% 判断是否提前终止迭代
       err1 = max(abs( f1{1}(:) - temp_f1{1}(:) ));
       err2 = max(abs( f1{2}(:) - temp_f1{2}(:) ));
       err3 = max(abs( f2{1}(:) - temp_f2{1}(:) ));
       err4 = max(abs( f2{2}(:) - temp_f2{2}(:) ));
       err5 = max(abs(Q{1}(:) - temp_Q{1}(:) ));
       err6 = max(abs(Q{2}(:) - temp_Q{2}(:) ));
       err7 = max(abs(S{1}(:) - temp_S{1}(:) ));
       err8= max(abs(S{2}(:) - temp_S{2}(:) ));
       err9 = max(abs(s(:) - temp_s(:) ));
                
       max_err = max( err1 , err2 );
       max_err = max( max_err , err3 );
       max_err = max( max_err , err4 );
       max_err = max( max_err , err5 );
       max_err = max( max_err , err6 );
       max_err = max( max_err , err7 );
       max_err = max( max_err , err8 );
       max_err = max( max_err , err9 );
  
       ERROR = [ERROR; max(max_err)];
       if(max_err<1e-4)
           break;
       end
       mu=min(max_mu,mu*1.5);
       for v=1:3
           y1{v}=y1{v}+mu*(q1{v}-f1{v});
           y2{v}=y2{v}+mu*(q2{v}-f2{v});
       end
       
       
       %% 保留上一次的r1,r2,fsal1,fsal2
       iter=iter+1;
       temp_f1 =f1;
       temp_f2 =f2;
       temp_Q=Q;
       temp_s=s;
       temp_S=S;
   
   end %%结束迭代
% plot(ERROR);
end %%结束CMR函数