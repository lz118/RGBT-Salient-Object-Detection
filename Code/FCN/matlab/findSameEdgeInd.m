function         index = findSameEdgeInd(edges,edges8);
num = length(edges8);
index = zeros(num,1);

for i = 1 : num
    ind1 = find(edges(:,1)==edges8(i,1));
    ind2 = find(edges(:,2)==edges8(i,2));
    index(i) = intersect(ind1,ind2);
end