

function [IM_B,subw]= myDrawBounderis(superpixel_labels,im)

% input_im = (imread(input_im));
[m,n,~] = size(im);
su = filter2(fspecial('average',3),superpixel_labels);
bw = su - superpixel_labels;
bw(abs(bw)<1e-4) = 0;
bw(1:end,1) = 0;
bw(1:end,end) = 0;
bw(1,1:end) = 0;
bw(end,1:end) = 0;
bw(bw~=0) = 1;
index = find(bw == 1);
inner_index = setdiff([1:m*n],index);
subw = superpixel_labels;
subw(inner_index) = 0;
imR = im(1:m,1:n,1);
imG = im(1:m,1:n,2);
imB = im(1:m,1:n,3);
imR(index) = 255;
imG(index) = 0;
imB(index) = 0;
IM_B = cat(3,imR,imG,imB);
% imwrite(IM_B,'segim.png');