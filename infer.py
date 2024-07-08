import cv2
import torch
from tqdm import tqdm
import os
from model import WavePaint
from datasets import make_default_val_dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

###########################################################################################################################################

base_path="WavePaint_"

NUM_MODULES 	= 8
NUM_BLOCKS		= 4 
MODEL_EMBEDDING	= 128 


model = WavePaint(
	num_modules			= NUM_MODULES,
	blocks_per_module 	= NUM_BLOCKS,
	mult 				= 4,
	ff_channel 			= MODEL_EMBEDDING,
	final_dim 			= MODEL_EMBEDDING,
	dropout 			= 0.5
).to(device)

PATH = base_path + '_blocks'+str(NUM_BLOCKS)+'_dim'+str(MODEL_EMBEDDING)+'_modules'+str(NUM_MODULES)+'_celebhq256.pth'

print(PATH)
model.load_state_dict(torch.load(PATH))
print("LOADED GEN WEIGHTS!!!")
model.eval()

indir = "/workspace/celebhq/val_256/random_thin_256/"
outdir = "/workspace/output/output/"
outdir2 = "/workspace/output/masked/"
out_ext = ".png"

dataset = make_default_val_dataset(indir, **{'kind': 'default', 'img_suffix': '.png', 'pad_out_to_modulo': 8})
print(len(dataset))

for img_i, data in tqdm(enumerate(dataset)):
	mask_fname = dataset.mask_filenames[img_i]
	cur_out_fname = outdir + os.path.splitext(mask_fname[len(indir):])[0] + out_ext

	cur_out_fname2 = outdir2 + os.path.splitext(mask_fname[len(indir):])[0] + out_ext
	os.makedirs(os.path.dirname(cur_out_fname), exist_ok=True)
	os.makedirs(os.path.dirname(cur_out_fname2), exist_ok=True)
	img, mask=torch.Tensor(data["image"]),torch.Tensor(data["mask"])
	h,w=img.shape[2], img.shape[2]
	ground_truth=img.clone().detach()
	img[:, :, :] = img[:, :, :] * (1-mask)
	masked_img=img
	
	out=model.forward((masked_img.reshape(-1,3,h,w)).to(device), mask.reshape(-1,1,h,w).to(device))

	cv2.imwrite(cur_out_fname,cv2.cvtColor(out[0].permute([1,2,0]).cpu().detach().numpy()*255, cv2.COLOR_RGB2BGR))
	cv2.imwrite(cur_out_fname2,cv2.cvtColor(masked_img.permute([1,2,0]).cpu().detach().numpy()*255, cv2.COLOR_RGB2BGR))
