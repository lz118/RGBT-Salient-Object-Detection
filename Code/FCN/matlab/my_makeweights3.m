function weights=my_makeweights3(edges,vals1,vals2,vals3,valScale,alpha)

valDistances1 = Dist(vals1,edges,'L2');
valDistances2 = Dist(vals2,edges,'L2');
valDistances3 = Dist(vals3,edges,'L2');

%%
valDistances = alpha(1)*valDistances1 + alpha(2)*valDistances2 + alpha(3)*valDistances3;
% valDistances = sqrt(valDistances);
valDistances = normalize(valDistances);
weights=exp(-valScale*valDistances);
%%
% weights1 = exp(-valScale*valDistances1);
% weights2 = exp(-valScale*valDistances2);
% weights = weights1*alpha + weights2;
% weights = (alpha+weights1) .* weights2;
% weights = normalize(weights);