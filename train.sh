python train.py \
       	--gpu_ids "0" \
	--batchSize 1 \
	--name Fashionade_TOP_v2 \
	--niter 25 \
	--niter_decay 25 \
	--dataroot ./dataset/Fashionade_train_sample \
	--dataset_mode viton \
	--no_flip \
	--preprocess_mode none \
	--label_nc 9 \
	--ngf 16 \
	--ndf 16 \
	#--num_D 1 #/data/meer/ClothWarping/PF-AFN/PF-AFN_train/results/PBAFN_fashionade_fashionadetop_0708 \

