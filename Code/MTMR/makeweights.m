function weights=makeweights(edges,vals,valScale)
valDistances=sqrt(sum((vals(edges(:,1),:)-vals(edges(:,2),:)).^2,2));
valDistances=normalize(valDistances); %Normalize to [0,1]
weights=exp(-valScale*valDistances);
