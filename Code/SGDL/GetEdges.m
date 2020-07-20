function my_edges = GetEdges(seed_fore,seed_back)

ind_fore = find(seed_fore==1);
ind_back = find(seed_back==1);
my_edges = [];
edges = zeros(length(ind_back),2);
edges(:,2) = ind_back;
for i = 1:length(ind_fore)
    edges(:,1) =  ind_fore(i);
    my_edges = [my_edges;edges];
end