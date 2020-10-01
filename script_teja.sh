
export PYTHONPATH=$PYTHONPATH:$(pwd)
CUDA_VISIBLE_DEVICES=0 python main.py --shared layer2 --rotation_type expand \
						--group_norm 8 \
                        --nepoch 150 --milestone_1 75 --milestone_2 125 \
                        --outf results/cifar10_layer2_gn_expand #>./txt_files/main.txt
 
CUDA_VISIBLE_DEVICES=0 python script_test_c10.py 5 layer2 slow gn_expand #>./txt_files/1.txt
#CUDA_VISIBLE_DEVICES=0 python script_test_c10.py 5 layer2 online gn_expand #>./txt_files/2.txt



#