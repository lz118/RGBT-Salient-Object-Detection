load('E:\experiment\cvpr13\multiM1\saliencymap\927\temp\test\mat\0.1.mat')
plot(Recall ,Pre, 'm-' ,'LineWidth', 2 )
axis([0 1 0 1])
xlabel('Recall')
ylabel('Precision')
hold on

load('E:\experiment\cvpr13\multiM1\saliencymap\927\temp\test\mat\0.5.mat')
plot(Recall ,Pre, 'b-' ,'LineWidth', 2 )
axis([0 1 0 1])
xlabel('Recall')
ylabel('Precision')
hold on

load('E:\experiment\cvpr13\multiM1\saliencymap\927\temp\test\mat\0.6.mat')
plot(Recall ,Pre, 'r-' ,'LineWidth', 2 )
axis([0 1 0 1])
xlabel('Recall')
ylabel('Precision')
hold on

load('E:\experiment\cvpr13\multiM1\saliencymap\927\temp\test\mat\1.mat')
plot(Recall ,Pre, 'y-' ,'LineWidth', 2 )
axis([0 1 0 1])
xlabel('Recall')
ylabel('Precision')
hold on

load('E:\experiment\cvpr13\multiM1\saliencymap\927\temp\test\mat\0.9.mat')
plot(Recall ,Pre, 'g-' ,'LineWidth', 2 )
axis([0 1 0 1])
xlabel('Recall')
ylabel('Precision')
hold on

% hleg=legend('Ours(G)','Ours(T)','Ours(GT)','Location','SouthWest');
% set(hleg,'FontSize',8)

