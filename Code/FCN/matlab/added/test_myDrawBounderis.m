

impath = '/media/iiau/linux_file/kyq/retrivel_saliency/ASD_codenew/figre/0271.bmp';
IM_B = myDrawBounderis(segim,impath);
seednum = length(seedlabel);
for ii = 1 : seednum
    ind = find(segim == seedlabel(ii));
    temp1 = IM_B(:,:,1);
    temp2 = IM_B(:,:,2);
    temp3 = IM_B(:,:,3);
    temp1(ind) = 255;
    temp2(ind) = 255;
    temp3(ind) = 0;
    IM_B = cat(3,temp1,temp2,temp3);
end
imshow(IM_B)
outname = './segsup.jpg';
imwrite(IM_B,outname);