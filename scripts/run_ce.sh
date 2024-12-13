# python utils.py get_save_dict_mean logs/cifar10_resnet50_cross_entropy_seed_{0,1,2}/H_W_colors_class_epoch_500.pkl --out_fname=mean.pkl

# CIFAR-10

## WideResNet-40x10 
python main.py --dataset=cifar10 --model=wide_resnet_40x10 --loss_fun=cross_entropy --lr=0.1 --seed=0 --device=cuda:0 &
python main.py --dataset=cifar10 --model=wide_resnet_40x10 --loss_fun=cross_entropy --lr=0.1 --seed=1 --device=cuda:1 &
wait
python main.py --dataset=cifar10 --model=wide_resnet_40x10 --loss_fun=cross_entropy --lr=0.1 --seed=2 --device=cuda:0 &
## ViT-B/4
python main.py --dataset=cifar10 --model=vit_b_4 --loss_fun=cross_entropy --lr=0.1 --seed=0 --device=cuda:1 &
wait
python main.py --dataset=cifar10 --model=vit_b_4 --loss_fun=cross_entropy --lr=0.1 --seed=1 --device=cuda:0 &
python main.py --dataset=cifar10 --model=vit_b_4 --loss_fun=cross_entropy --lr=0.1 --seed=2 --device=cuda:1 &
wait

# CIFAR-100

## WideResNet-40x10 
python main.py --dataset=cifar100 --model=wide_resnet_40x10 --loss_fun=cross_entropy --lr=0.1 --seed=0 --device=cuda:0 &
python main.py --dataset=cifar100 --model=wide_resnet_40x10 --loss_fun=cross_entropy --lr=0.1 --seed=1 --device=cuda:1 &
wait
python main.py --dataset=cifar100 --model=wide_resnet_40x10 --loss_fun=cross_entropy --lr=0.1 --seed=2 --device=cuda:0 &
## ViT-B/4
python main.py --dataset=cifar100 --model=vit_b_4 --loss_fun=cross_entropy --lr=0.1 --seed=0 --device=cuda:1 &
wait
python main.py --dataset=cifar100 --model=vit_b_4 --loss_fun=cross_entropy --lr=0.1 --seed=1 --device=cuda:0 &
python main.py --dataset=cifar100 --model=vit_b_4 --loss_fun=cross_entropy --lr=0.1 --seed=2 --device=cuda:1 &
wait

# Fashion-MNIST

## WideResNet-40x10 
python main.py --dataset=fashionmnist --model=wide_resnet_40x10 --loss_fun=cross_entropy --lr=0.1 --seed=0 --device=cuda:0 &
python main.py --dataset=fashionmnist --model=wide_resnet_40x10 --loss_fun=cross_entropy --lr=0.1 --seed=1 --device=cuda:1 &
wait
python main.py --dataset=fashionmnist --model=wide_resnet_40x10 --loss_fun=cross_entropy --lr=0.1 --seed=2 --device=cuda:0 &
## ViT-B/4
python main.py --dataset=fashionmnist --model=vit_b_4 --loss_fun=cross_entropy --lr=0.1 --seed=0 --device=cuda:1 &
wait
python main.py --dataset=fashionmnist --model=vit_b_4 --loss_fun=cross_entropy --lr=0.1 --seed=1 --device=cuda:0 &
python main.py --dataset=fashionmnist --model=vit_b_4 --loss_fun=cross_entropy --lr=0.1 --seed=2 --device=cuda:1
