import torch

model = WavePaint(
	num_modules	  = 8,
	blocks_per_module = 4,
	mult 		  = 4,
	ff_channel 	  = 128,
	final_dim 	  = 128,
	dropout 	  = 0.5
).to(device)

url = 'https://huggingface.co/cloudwalker/wavemix/resolve/main/Weights/WavePaint__blocks4_dim128_modules8_places256_medium_mask.pth' #Places dataset trained with medium masks
url = 'https://huggingface.co/cloudwalker/wavemix/resolve/main/Weights/WavePaint__blocks4_dim128_modules8_places256_thick_mask.pth'  #Places dataset trained with thick masks
url = 'https://huggingface.co/cloudwalker/wavemix/resolve/main/Weights/WavePaint_blocks4_dim128_modules8_celebhq_medium_mask.pth'    #CelebAHQ dataset trained with medium masks
url = 'https://huggingface.co/cloudwalker/wavemix/resolve/main/Weights/WavePaint_blocks4_dim128_modules8_celebhq_thick_mask.pth'     #CelebAHQ dataset trained with thick masks

model.load_state_dict(torch.hub.load_state_dict_from_url(url))
