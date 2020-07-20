clc
clear
figure;
MR=load('E:\WORK2\AAAI+IEEETran\AAAI compare data\1000saliencymap\RGBT\MR\out.mat')
plot(MR.Recall ,MR.Pre, 'c-' ,'LineWidth',2)
axis([0 1 0 1])
xlabel('Recall')
ylabel('Precision')
hold on

MSS=load('E:\WORK2\AAAI+IEEETran\AAAI compare data\1000saliencymap\RGBT\MSS\out.mat')
plot(MSS.Recall ,MSS.Pre, 'g--' ,'LineWidth', 2 )
axis([0 1 0 1])
xlabel('Recall')
ylabel('Precision')
hold on

RBD=load('E:\WORK2\AAAI+IEEETran\AAAI compare data\1000saliencymap\RGBT\RBD\out.mat')
plot(RBD.Recall ,RBD.Pre, 'b-' ,'LineWidth', 2 )
axis([0 1 0 1])
xlabel('Recall')
ylabel('Precision')
hold on

CA=load('E:\WORK2\AAAI+IEEETran\AAAI compare data\1000saliencymap\RGBT\BSCA\out.mat')
plot(CA.Recall ,CA.Pre, 'b-.' ,'LineWidth',2)
axis([0 1 0 1])
xlabel('Recall')
ylabel('Precision')
hold on

BL=load('E:\WORK2\AAAI+IEEETran\AAAI compare data\1000saliencymap\RGBT\BL\out.mat')
plot(BL.Recall ,BL.Pre, 'y-' ,'LineWidth', 2 )
axis([0 1 0 1])
xlabel('Recall')
ylabel('Precision')
hold on

RRWR=load('E:\WORK2\AAAI+IEEETran\AAAI compare data\1000saliencymap\RGBT\RRWR\out.mat')
plot(RRWR.Recall ,RRWR.Pre, 'g-' ,'LineWidth',2)
axis([0 1 0 1])
xlabel('Recall')
ylabel('Precision')
hold on

% FCNN=load('E:\WORK2\AAAI+IEEETran\AAAI compare data\1000saliencymap\RGBT\FCNN\out.mat')
% plot(FCNN.Recall ,FCNN.Pre, 'c-.' ,'LineWidth', 2 )
% axis([0 1 0 1])
% xlabel('Recall')
% ylabel('Precision')
% hold on

DSS=load('E:\WORK2\AAAI+IEEETran\AAAI compare data\1000saliencymap\RGBT\DSS\out.mat')
plot(DSS.Recall ,DSS.Pre, 'y-.' ,'LineWidth', 2 )
axis([0 1 0 1])
xlabel('Recall')
ylabel('Precision')
hold on

MTMR= load('E:\WORK2\AAAI+IEEETran\AAAI compare data\1000saliencymap\RGBT\MTMR\out.mat')
plot(MTMR.Recall ,MTMR.Pre, 'm-.' ,'LineWidth', 2 )
axis([0 1 0 1])
xlabel('Recall')
ylabel('Precision')
hold on


Old=load('E:\WORK1\2MIPR\our\1k\out.mat')
plot(Old.Recall ,Old.Pre, 'b-' ,'LineWidth',2)
axis([0 1 0 1])
xlabel('Recall')
ylabel('Precision')
hold on

Our=load('E:\WORK1\code\theta1=40;theta2=40;gama=1.5;\out.mat')
plot( Our.Recall , Our.Pre, 'r-' ,'LineWidth',2)
axis([0 1 0 1])
xlabel('Recall')
ylabel('Precision')
hold on

methods={'MR','MSS','RBD','CA','BL','RRWR','DSS','MTMR','M3S-NIR','Our'};
precisions = {MR.FMeasureF, MSS.FMeasureF, RBD.FMeasureF, CA.FMeasureF, BL.FMeasureF, RRWR.FMeasureF, DSS.FMeasureF, MTMR.FMeasureF,Old.FMeasureF, Our.FMeasureF};




for i  = 1:size(methods, 2)
    legendlabel{i} = [methods{i} ' [' sprintf('%.3f', precisions{i}) ']'];
end
legend(legendlabel, 'Location', 'SouthWest', 'FontSize', 13)

