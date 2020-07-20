function subw = findSupBoundaries(sulabel)

[m,n] = size(sulabel);
su = filter2(fspecial('average',3),sulabel);
bw = su - sulabel;
bw(abs(bw)<1e-4) = 0;
bw(1:end,1) = 0;
bw(1:end,end) = 0;
bw(1,1:end) = 0;
bw(end,1:end) = 0;
bw(bw~=0) = 1;
index = find(bw == 1);
inner_index = setdiff([1:m*n],index);
subw = sulabel;
subw(inner_index) = 0;
