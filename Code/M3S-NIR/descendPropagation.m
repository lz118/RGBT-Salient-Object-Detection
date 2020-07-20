function propagatedData = descendPropagation(feat,initData,Nsample,featDim)
%% Calculate the propagated reconstruction errors.
%% Input:
%Nsample:the number of supixel.
% feat: superpixel feature matrix, the size of which is Nsample * 3.
% initData: the initial saliency .
% paramPropagate: propagation parameters.
%paramPropagate.nclus = 8;   [the number of clusters]
%paramPropagate.maxIter=200;  Max number of k-means iterations
%% Output:
% propagatedData: the propagated saliency.   
%%%%%%%%%%%%%%%%
paramPropagate.nclus = 8;
paramPropagate.maxIter=200;
paramPropagate.lamna=0.5;
%% calculate the cluster centers 
centers = form_codebook(feat', paramPropagate.nclus,paramPropagate.maxIter);
%% label each superpixel according to the cluster centers.
[ featLabel ] = labelCluster( centers, feat', Nsample,  paramPropagate.nclus ); 
%% a row vector(1*3),each element is the average of the column of feature
meanfeat = mean(feat,1); 
%% sum of the variance of each feature %%
sig2 = zeros(1,featDim); %a row vector
for k=1:featDim
    
    sig2(k) = norm(feat(:,k) - meanfeat(:,k))^2/Nsample;
end
%sigma2 = mean(sig2);
sigma2 = featDim^2\(sum(sig2));
%% Calculate the Euclidean distance matrix (the diag element is zero)
distMatrix = zeros(Nsample, Nsample);
for i=1:Nsample
    for j=i+1:Nsample
        distMatrix(i,j) = exp(-norm(feat(i,:)-feat(j,:))^2/(2*sigma2));
        distMatrix(j,i) = distMatrix(i,j);
    end
end

%% Propagation
[desData desInd] = sort(initData); 
for i=Nsample:-1:1
    dataLabel = desInd(i);
    clusterlabel = featLabel(dataLabel);
    clusterbgsup = find(featLabel==clusterlabel);
    nInnerCluster = length(clusterbgsup);

	sumdist = 0;
	sumA = 0;
	for m=1:nInnerCluster
        M = clusterbgsup(m);
        sumdist = sumdist + distMatrix(dataLabel,M)*initData(M);
        sumA = sumA + distMatrix(dataLabel,M);
	end
    
    if sumA==0
        sumA = sumA+eps;
    end
	initData(dataLabel)=(1-paramPropagate.lamna)*initData(dataLabel) + paramPropagate.lamna/sumA*sumdist;
end
propagatedData = initData;
propagatedData = (propagatedData - min(propagatedData(:)))/(max(propagatedData(:)) - min(propagatedData(:)));