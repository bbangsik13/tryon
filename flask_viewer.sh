# test result 
echo http://125.6.38.24:8005/
#python flask_viewer.py 8009 checkpoints/mpv_half/web/images
# as same as test with bs = 1 
#python flask_viewer.py 8009 checkpoints/mpv_half_1/web/images
# as same as test bs = 4
#python flask_viewer.py 8009 checkpoints/mpv_half_bs4/web/images
# copying the Wc onto Igt
#python flask_viewer.py 8009 checkpoints/mpv_half_2/web/images
# using Wc = Igt[Wcm] 
#ETRI bot
#python flask_viewer.py 8010 checkpoints/ETRI-Bot_new_L1/web/images
#MPV TOP mfpng
#python flask_viewer.py 8010 checkpoints/ALIAS_ETRI_BOT_clothflow_vis_mask_v3/web/images
#python flask_viewer.py 8006 checkpoints/ALIAS_ETRI_TOP_pbafn_mask_vis_v2/web/images
#python flask_viewer.py 8006 checkpoints/ALIAS_ETRI_BOT_pbafn_swap_25epoch_v3/web/images
#python flask_viewer.py 8006 checkpoints/ALIAS_ETRI_TOP_pbafn_swap_50epoch_v3/web/images 
#python flask_viewer.py 8010 checkpoints/ALIAS_ETRI_TOP_pbafn_swap_50epoch/web/images
# python flask_viewer.py 8005 checkpoints/Fashionade_TOP_v5/web/images
python flask_viewer.py 8005 results/baek/val_latest/images/synthesized_image # masked/ 
#python flask_viewer.py 8006 results/Fashionade_TOP/val_latest/images/masked/

