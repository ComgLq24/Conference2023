#############################Train Deeplabv3 on CaDIS###########################################
python train.py \
\
--dataroot ./datasets/CaDIS_clean/ \
--name Deeplabv3_resnet101_CaDIS_merge_tools \
--gpu_ids 0 \
--checkpoints_dir ./checkpoints/source_domain \
--model deeplabv3 \
--input_nc 3 \
--output_nc 7 \
--netTask deeplabv3_resnet101 \
--pretrained \
--dataset_mode source_only \
--shuffle True \
--num_threads 8 \
--batch_size 3 \
--load_size 720 \
--preprocess rescale_rotate_flip \
--mapping_file_name mapping_merge_tools \
--drop_last \
--eval \
--description "deeplabv3 model without pretrained" \
--add_timestamp \
\
--display_freq 40 \
--display_ncols 3 \
--display_env Deeplabv3_resnet101_CaDIS_merge_tools \
--display_port 8090 \
--save_epoch_freq 5 \
--n_epochs 80 \
--n_epochs_decay 40 \
--lr 0.0005 \
--lr_policy linear \
--ignore_label 6 \
--validate \
--validation_dataset_mode source_only \
--validation_dataset_root ./datasets/CaDIS_clean/ \
--validation_preprocess rescale \
--validation_load_size 720 \
--validation_batch_size 3

#############################Test Deeplabv3 on CaDIS###########################################
python test.py \
\
--dataroot ./datasets/CaDIS_clean/ \
--name Deeplabv3_resnet101_CaDIS_merge_tools \
--gpu_ids 0 \
--checkpoints_dir ./checkpoints/source_domain \
--model deeplabv3 \
--input_nc 3 \
--output_nc 7 \
--netTask deeplabv3_resnet101 \
--dataset_mode source_only \
--shuffle False \
--num_threads 8 \
--batch_size 3 \
--load_size 720 \
--preprocess rescale \
--epoch 120 \
--mapping_file_name mapping_merge_tools \
--eval \
\
--results_dir ./results/CaDIS_clean \
--num_test 2000 

python ./metrics.py \
--results_dir ./results/CaDIS_clean/Deeplabv3_resnet101_CaDIS_merge_tools \
--epoch 20