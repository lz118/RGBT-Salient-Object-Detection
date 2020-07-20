function B = ReadDAT(image_size,data_path)
% $Description:
%    -read the superpixel labels from .dat file
% $Agruments
% Input;
%    -image_size: [width height]
%    -data_path: the path of the .dat file 
% Output:
%    -label matrix width*height

row = image_size(1);
colomn = image_size(2);
fid = fopen(data_path,'r');
A = fread(fid, row * colomn, 'uint32')';
A = A + 1;
B = reshape(A,[colomn, row]);
B = B';
fclose(fid);