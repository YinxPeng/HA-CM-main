export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0

# NTU60 xsub
python -m torch.distributed.launch --nproc_per_node=1 --master_port 11234 main_pretrain.py \
--config ./config/ntu60_xsub_joint/pretrain_hacm.yaml \
--output_dir ./output_dir/ntu60_xsub_joint/pretrain \
--log_dir ./output_dir/ntu60_xsub_joint/pretrain

# NTU60 xview
#python -m torch.distributed.launch --nproc_per_node=4 --master_port 11234 main_pretrain.py \
#--config ./config/ntu60_xview_joint/pretrain_hacm.yaml \
#--output_dir ./output_dir/ntu60_xview_joint/pretrain \
#--log_dir ./output_dir/ntu60_xview_joint/pretrain
#
## NTU120 xset
#python -m torch.distributed.launch --nproc_per_node=4 --master_port 11234 main_pretrain.py \
#--config ./config/ntu120_xset_joint/pretrain_hacm.yaml \
#--output_dir ./output_dir/ntu120_xset_joint/pretrain \
#--log_dir ./output_dir/ntu120_xset_joint/pretrain
#
## NTU120 xsub
#python -m torch.distributed.launch --nproc_per_node=4 --master_port 11234 main_pretrain.py \
#--config ./config/ntu120_xsub_joint/pretrain_hacm.yaml \
#--output_dir ./output_dir/ntu120_xsub_joint/pretrain \
#--log_dir ./output_dir/ntu120_xsub_joint/pretrain
#
## PKU v1
#python -m torch.distributed.launch --nproc_per_node=4 --master_port 11234 main_pretrain.py \
#--config ./config/pkuv1_xsub_joint/pretrain_hacm.yaml \
#--output_dir ./output_dir/pkuv1_xsub_joint/pretrain \
#--log_dir ./output_dir/pkuv1_xsub_joint/pretrain
#
## PKU v2
#python -m torch.distributed.launch --nproc_per_node=4 --master_port 11234 main_pretrain.py \
#--config ./config/pkuv2_xsub_joint/pretrain_hacm.yaml \
#--output_dir ./output_dir/pkuv2_xsub_joint/pretrain \
#--log_dir ./output_dir/pkuv2_xsub_joint/pretrain
