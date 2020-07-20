function Y = ComputeAffinityMatrix(L1,L2,spnum)

gama = 8;
mu = 0.001;
bata1 = 0.5;
bata2 = 0.5;
jjj=0;
error = 1;
while((jjj<5)&&(error>0.0001))
    jjj = jjj+1;
    % 固定bata,求最优的Y
    Y = inv((bata1^gama).*L1+(bata2^gama).*L2+mu*eye(spnum));
    % 固定Y,求最优的bata
    mm1 = (trace(Y'*L1*Y))^(1/(1-gama));
    mm2 = (trace(Y'*L2*Y))^(1/(1-gama));
    bata1_new = mm1/(mm1+mm2);
    bata2_new = mm2/(mm1+mm2);

    error = abs(bata1-bata1_new);
    bata1=bata1_new;
    bata2=bata2_new;
end