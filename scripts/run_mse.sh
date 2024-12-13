# CIFAR-10

## WideResNet-40x10 
python main.py --dataset=cifar10 --model=wide_resnet_40x10 --loss_fun=scaled_mse --lr=0.1 --seed=0 --device=cuda:0
python main.py --dataset=cifar10 --model=wide_resnet_40x10 --loss_fun=scaled_mse --lr=0.1 --seed=1 --device=cuda:1 &
wait
python main.py --dataset=cifar10 --model=wide_resnet_40x10 --loss_fun=scaled_mse --lr=0.1 --seed=2 --device=cuda:0 &
## ViT-B/4
python main.py --dataset=cifar10 --model=vit_b_4 --loss_fun=scaled_mse --lr=0.1 --seed=0 --device=cuda:1 &
wait
python main.py --dataset=cifar10 --model=vit_b_4 --loss_fun=scaled_mse --lr=0.1 --seed=1 --device=cuda:0 &
python main.py --dataset=cifar10 --model=vit_b_4 --loss_fun=scaled_mse --lr=0.1 --seed=2 --device=cuda:1 &
wait

# CIFAR-100

## WideResNet-40x10 
python main.py --dataset=cifar100 --model=wide_resnet_40x10 --loss_fun=scaled_mse --lr=0.1 --seed=0 --device=cuda:0 &
python main.py --dataset=cifar100 --model=wide_resnet_40x10 --loss_fun=scaled_mse --lr=0.1 --seed=1 --device=cuda:1 &
wait
python main.py --dataset=cifar100 --model=wide_resnet_40x10 --loss_fun=scaled_mse --lr=0.1 --seed=2 --device=cuda:0 &
## ViT-B/4
python main.py --dataset=cifar100 --model=vit_b_4 --loss_fun=scaled_mse --lr=0.1 --seed=0 --device=cuda:1 &
wait
python main.py --dataset=cifar100 --model=vit_b_4 --loss_fun=scaled_mse --lr=0.1 --seed=1 --device=cuda:0 &
python main.py --dataset=cifar100 --model=vit_b_4 --loss_fun=scaled_mse --lr=0.1 --seed=2 --device=cuda:1 &
wait

# Fashion-MNIST

## WideResNet-40x10 
python main.py --dataset=fashionmnist --model=wide_resnet_40x10 --loss_fun=scaled_mse --lr=0.1 --seed=0 --device=cuda:0 &
python main.py --dataset=fashionmnist --model=wide_resnet_40x10 --loss_fun=scaled_mse --lr=0.1 --seed=1 --device=cuda:1 &
wait
python main.py --dataset=fashionmnist --model=wide_resnet_40x10 --loss_fun=scaled_mse --lr=0.1 --seed=2 --device=cuda:0 &
## ViT-B/4
python main.py --dataset=fashionmnist --model=vit_b_4 --loss_fun=scaled_mse --lr=0.1 --seed=0 --device=cuda:1 &
wait
python main.py --dataset=fashionmnist --model=vit_b_4 --loss_fun=scaled_mse --lr=0.1 --seed=1 --device=cuda:0 &
python main.py --dataset=fashionmnist --model=vit_b_4 --loss_fun=scaled_mse --lr=0.1 --seed=2 --device=cuda:1
