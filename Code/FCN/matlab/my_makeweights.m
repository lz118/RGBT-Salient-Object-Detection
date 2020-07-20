function weights=my_makeweights(edges,vals1,vals2,valScale,alpha,edges_weights)
if nargin <6
    edges_weights = 1;
end

valDistances1 = Dist(vals1,edges,'L2');
valDistances2 = Dist(vals2,edges,'L2');
%%
valDistances = alpha(1)*valDistances1 + alpha(2)*valDistances2;
% valDistances = valDistances1.*(valDistances2+1);
valDistances = normalize(valDistances).*edges_weights;
weights=exp(-valScale*valDistances);
%%
% weights1 = exp(-valScale*valDistances1);
% weights2 = exp(-valScale*valDistances2);
% weights = weights1*alpha + weights2;
% weights = (alpha+weights1) .* weights2;
% weights = normalize(weights);