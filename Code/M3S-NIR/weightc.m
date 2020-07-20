function   [c]=weightc(index,super,pixlist)
for v=1:3
      c{v} = zeros(index{2},index{v});%200*100,200*200,200*300
        for k = 1:index{2}
               uniq = unique(super{v}(pixlist{2}{k}));%第二个有多少个位置200
            for j = 1:size(uniq,1)
                c{v}(k, uniq(j)) = length(find(super{v}(pixlist{2}{k})==uniq(j)))/size(pixlist{2}{k},1);
            end
        end
end