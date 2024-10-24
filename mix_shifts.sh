model_name="resnet18"
# model_name="mobilenetv2"
# for method in tent eata cotta
# do 
# 	for bs in 64 16 1
# 	do
# 		python3 run_all_cifar10.py --exp_type="mix_shifts" --tta=$method --batch_size=$bs --model=$model_name --data_size=100 --save --precision="fp32" --use_cuda --seed=$1 --class_num=$3
# 	done
# done

# python3 run_all_cifar10.py --exp_type="mix_shifts" --tta="none" --batch_size=1 --model=$model_name --data_size=100 --precision="fp32" --seed=$1 --class_num=$3 --save
python3 run_all_cifar10.py --exp_type="mix_shifts" --tta="quantnone" --batch_size=1 --model=$model_name --data_size=100 --precision="int8" --seed=$1 --class_num=$3 --save
# python3 run_all_cifar10.py --exp_type="mix_shifts" --tta="ours" --batch_size=1 --model=$model_name --data_size=100  --precision="fp32" --seed=$1 --class_num=$3 --save
# python3 run_all_cifar10.py --exp_type="mix_shifts" --tta="quantours" --batch_size=1 --model=$model_name --data_size=100 --precision="int8" --seed=$1 --class_num=$3 --save
# python3 run_all_cifar10.py --exp_type="mix_shifts" --tta="tema" --batch_size=1 --model=$model_name --data_size=100 --precision="fp32" --seed=$1 --use_cuda --class_num=$3 --save
# mv mix_shifts_cifar$3_dset.pth mix_shifts_cifar$3_dset$1.pth
# mv mix_shifts_cifar$3_dset$2.pth mix_shifts_cifar$3_dset.pth
