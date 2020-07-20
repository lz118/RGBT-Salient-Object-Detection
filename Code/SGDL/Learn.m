function [s] = Learn(G_L1,G_L2,G_L3,T_L1,T_L2,T_L3,spnum,q)
I=eye(spnum);
gama1 =8;%%%层之间
gama2 =0.5;%%%模态之间
theta=0.0001;
mu = 0.001;%%%%
lambda1=0.004;%%lambda/theta;
G_beta1 = 1/3;
temp_G_beta1=1/3;
G_beta2 = 1/3;
temp_G_beta2=1/3;
G_beta3=1/3;
T_beta1 = 1/3;
temp_T_beta1=1/3;
T_beta2 = 1/3;
temp_T_beta2=1/3;
T_beta3 = 1/3;
G_alpha=0.5;
temp_G_alpha=0.5;
T_alpha=0.5;
iter=1; 
maxIter=50;
s=zeros(spnum,1);
temp_s=s;
 ERROR = [];
while(iter<=maxIter )
 %% 求learned W
 for i=1:spnum
    for j=1:spnum
    a(i,j)=(s(i)-s(j))^2;
    end
end
G_L=(G_beta1^gama1).*G_L1+(G_beta2^gama1).*G_L2+(G_beta3^gama1).*G_L3;
T_L=(T_beta1^gama1).*T_L1+(T_beta2^gama1).*T_L2+(T_beta3^gama1).*T_L3;
L=(G_alpha^gama2).*G_L+(T_alpha^gama2).*T_L;
 optAff=inv(L+mu*I);
W=optAff*(mu*I-theta*1/4*a);
W(W<0)=0;
           %W=optAff;
  %% 固定 W,求可见光层之间的最优的beta
    G_mm1 = (trace(W'*G_L1*W))^(1/(1-gama1));
    G_mm2 = (trace(W'*G_L2*W))^(1/(1-gama1));
    G_mm3 = (trace(W'*G_L3*W))^(1/(1-gama1));
    G_beta1= G_mm1/(G_mm1+G_mm2+G_mm3);
    G_beta2 = G_mm2/(G_mm1+G_mm2+G_mm3);
    G_beta3 = G_mm3/(G_mm1+G_mm2+G_mm3);
   %% 固定W,求红外层之间的最优的beta 
    T_mm1 = (trace(W'*T_L1*W))^(1/(1-gama1));
    T_mm2 = (trace(W'*T_L2*W))^(1/(1-gama1));
     T_mm3 = (trace(W'*T_L3*W))^(1/(1-gama1));
    T_beta1 = T_mm1/(T_mm1+T_mm2+T_mm3);
    T_beta2= T_mm2/(T_mm1+T_mm2+T_mm3);
   T_beta3= T_mm3/(T_mm1+T_mm2+T_mm3);
 %% 求两个模态间的权重alpha
   G_mm = ((G_beta1^gama1)*trace(W'*G_L1*W)+(G_beta2^gama1)*trace(W'*G_L2*W)+(G_beta3^gama1)*trace(W'*G_L3*W))^(1/(1-gama2));
   T_mm = ((T_beta1^gama1)*trace(W'*T_L1*W)+(T_beta2^gama1)*trace(W'*T_L2*W)+(T_beta3^gama1)*trace(W'*T_L3*W))^(1/(1-gama2));  
   G_alpha = G_mm/(G_mm+T_mm);
   T_alpha = T_mm/(G_mm+T_mm);
  %% 求s
 dd = sum(W); D = sparse(1:spnum,1:spnum,dd); clear dd;
 F=D-W; 
 s= inv(F/lambda1+I)*q; 
 s(s<0)=0;
             %% 判断是否提前终止迭代
        err1 = abs(G_beta1-temp_G_beta1); 
        err2 = abs(G_beta2-temp_G_beta2); 
         err3 = abs(T_beta1-temp_T_beta1); 
         err4 = abs(T_beta2-temp_T_beta2); 
        err5 = abs(G_alpha-temp_G_alpha);
        err6 = max(abs( s(:) - temp_s(:) ));
          max_err = max( err1 , err2 );
          max_err = max( max_err , err3 );  
          max_err = max( max_err , err4 );  
           max_err = max( max_err , err5 );  
            max_err = max( max_err , err6 );  
          ERROR = [ERROR; max(max_err)];  
       if(max_err<1e-4)
         break;
       end
       %% 保留上一次的result
      iter=iter+1;
      temp_G_beta1 =G_beta1;
       temp_G_beta2 =G_beta2;
       temp_T_beta1=T_beta1;
        temp_T_beta2=T_beta2;
       temp_G_alpha=G_alpha;
       temp_s=s;
end