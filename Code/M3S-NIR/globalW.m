function adjMergeW = globalW(spnum)
%% global constructing graph W

adjMergeW=ones(spnum,spnum);
for i=1:spnum
    adjMergeW(i, i)=0;
end



