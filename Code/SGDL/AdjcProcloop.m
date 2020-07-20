function adjcMerge = AdjcProcloop(M,N)
% $Description:
%    -compute the adjacent matrix
% $Agruments
% Input;
%    -M: superpixel label matrix
%    -N: superpixel number 
% Output:
%    -adjcMerge: adjacent matrix

adjcMerge = zeros(N,N);
[m n] = size(M);

for i = 1:m-1
    for j = 1:n-1
        if(M(i,j)~=M(i,j+1))
            adjcMerge(M(i,j),M(i,j+1)) = 1;
            adjcMerge(M(i,j+1),M(i,j)) = 1;
        end;
        if(M(i,j)~=M(i+1,j))
            adjcMerge(M(i,j),M(i+1,j)) = 1;
            adjcMerge(M(i+1,j),M(i,j)) = 1;
        end;
        if(M(i,j)~=M(i+1,j+1))
            adjcMerge(M(i,j),M(i+1,j+1)) = 1;
            adjcMerge(M(i+1,j+1),M(i,j)) = 1;
        end;
        if(M(i+1,j)~=M(i,j+1))
            adjcMerge(M(i+1,j),M(i,j+1)) = 1;
            adjcMerge(M(i,j+1),M(i+1,j)) = 1;
        end;
    end;
end;    
bd=unique([M(1,:),M(m,:),M(:,1)',M(:,n)']);
for i=1:length(bd)
    for j=i+1:length(bd)
        adjcMerge(bd(i),bd(j))=1;
        adjcMerge(bd(j),bd(i))=1;
    end
end
    