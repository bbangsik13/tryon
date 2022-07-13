python train.py \
       	--gpu_ids "2,3" \
	--batchSize 2 \
	--name Fashionade_TOP \
	--niter 25 \
	--niter_decay 25 \
	--dataroot /data/meer/ClothWarping/PF-AFN/PF-AFN_train/results/PBAFN_fashionade_fashionadetop_0708 \
	--dataset_mode viton \
	--no_flip \
	--preprocess_mode none \
	--label_nc 14 \
	--ngf 16 \
	--ndf 16 \
	#--num_D 1

