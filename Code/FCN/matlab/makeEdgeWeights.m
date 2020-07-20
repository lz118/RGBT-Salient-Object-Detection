function    edges_weights = makeEdgeWeights(edges,imedge,SupBound,theta)
num = length(edges);
IND = findneighboredge(edges,SupBound);
edge_grade = zeros(num,1);
for i = 1 : num
    edge_grade(i) = sum(imedge(IND{i}));
end
ind0 = find(edge_grade==0);
ind = setdiff([1:num],ind0);
edge_grade(ind) = normalize(edge_grade(ind));
edges_weights = edge_grade;
% edges_weights=exp(-theta*edge_grade);
% edges_weights = 1./(edge_grade+1);

