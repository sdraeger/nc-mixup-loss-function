declare -a loss_funs=("cross_entropy" "focal_loss" "scaled_mse" "mae")
declare -a nets=("wide_resnet_40x10" "vit_b_4")
declare -a datasets=("cifar10" "fashionmnist")

for loss_fun in "${loss_funs[@]}"; do
	for net in "${nets[@]}"; do
		for dataset in "${datasets[@]}"; do
			echo "${dataset}_${net}_${loss_fun}"

			python utils.py plot_last_layer \
				--pkl_dict_fname=logs/${dataset}_${net}_${loss_fun}/H_W_colors_class_epoch_1_mean.pkl \
				--epoch=1 \
				--out_filename=logs/${dataset}_${net}_${loss_fun}/last_layer.pdf

			python utils.py plot_last_layer \
				--pkl_dict_fname=logs/${dataset}_${net}_${loss_fun}/H_W_colors_class_epoch_500_mean.pkl \
				--epoch=500 \
				--out_filename=logs/${dataset}_${net}_${loss_fun}/last_layer.pdf
		done
	done
done
