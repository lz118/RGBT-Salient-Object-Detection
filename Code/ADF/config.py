#coding=utf-8
#for net
n_color=3
cuda=True
GPUid=2
mode="test"
useMSA=True
useCA=True
pretrained_model='vgg16.pth'
save_folder="./model"
edge_loss=True
config_vgg = {'deep_pool': [[512, 512, 256, 128], [512, 256, 128, 128], [True, True, True, False], [True, True, True, False]], 'score': 128}


#for train
lr=1e-4
wd=0.0005
epoch=25
lr_decay_epoch=[]
batch_size=1
num_thread=1
epoch_save=5
iter_size=10
show_every=100

train_root="/"

#for test
model='model/final.pth'
test_root='/'
test_fold="./our/"
