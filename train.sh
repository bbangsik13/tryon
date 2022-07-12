python train.py \
       	--gpu_ids "0" \
	--batchSize 1 \
	--name Fashionade_train_sample \
	--niter 100 \
	--niter_decay 100 \
	--dataroot ./dataset/Fashionade_train_sample \
	--dataset_mode viton \
	--no_flip \
	--preprocess_mode none \
	--label_nc 14 \
	--ngf 16 \
	--ndf 16 \
	#--num_D 1

