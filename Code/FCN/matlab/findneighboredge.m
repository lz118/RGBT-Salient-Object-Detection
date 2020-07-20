function IND = findneighboredge(edges,SupBound)
[m,n] = size(SupBound);
num = length(edges);
IND = cell(num,1);
for i = 1 : num
%     disp(i);
    temp = edges(i,:);
    index1 = SupBound==temp(1);
    index2 = SupBound==temp(2);
    SupBound1 = zeros(m,n);
    SupBound1(index1) = temp(1);
    SupBound2 = zeros(m,n);
    SupBound2(index2) = temp(2);
    % index = intersect(index1,index2);
    % SupBound(index) = 0;
    se = strel('disk',1);
    SupBound1 = imdilate(SupBound1,se);
    SupBound2 = imdilate(SupBound2,se);
    SupBound3 = SupBound1 + SupBound2;
    index = find(SupBound3 == sum(temp(:)));
%     ww = zeros(m,n);
%     ww(index)=1;
    IND{i} = index;
end