function [idxImg,adjcMatrix8, pixelList] = SLIC_Split(noFrameImg, spnumber, spPath, srcName)
% Segment rgb image into super-pixels using SLIC algorithm:

% R.Achanta, A.Shaji, K.Smith, A.Lucchi, P.Fua, and S.Susstrunk. Slic 
% superpixels compared to state-of-the-art superpixel methods. IEEE
% Transactions on Pattern Analysis and Machine Intelligence, 2012.

% Author: Wangjiang Zhu
% Email: wangjiang88119@gmail.com
% Date: 3/24/2014

%% Segment using SLIC:
compactness = 20;   %the larger compactness is, the more regular superpixels will be
% The logic in SLIC_mex:
% 1.If spPath or srcName are NOT specified:
%   SLIC_mex run segmentation algorithm
% 2.If spPath and srcName are specified:
%   SLIC_mex will check if index image file is already exists under
%   spPath:
%   2.1 If so, just load it.
%   2.2 If not, run segmentation algorithm, save index image and mean color
%   image
noFrameImg = uint8(noFrameImg*255);
if 2 == nargin
    [idxImg, spNum] = SLIC_mex(noFrameImg, spnumber, compactness);
elseif 4 == nargin
    [idxImg, spNum] = SLIC_mex(noFrameImg, spnumber, compactness, spPath, srcName);
else
    error('wrong input para number');
end
%%
adjcMatrix8 = AdjcProcloop(idxImg, spNum);

%%
pixelList = cell(spNum, 1);
for n = 1:spNum
    pixelList{n} = find(idxImg == n);
end
