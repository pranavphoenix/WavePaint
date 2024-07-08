import torch, torchvision
import torch.nn as nn
import numpy as np
import cv2, sys, os, time, lpips, argparse 
from tqdm import tqdm
from torchinfo import summary
from torchmetrics.functional import peak_signal_noise_ratio
from torchmetrics.functional import structural_similarity_index_measure
from datasets import make_default_train_dataloader, make_default_val_dataloader
from model import WavePaint
import torch.optim as optim
parser = argparse.ArgumentParser()

parser.add_argument("-mask", "--mask", default="medium", help = "Mask size: thick or medium")
parser.add_argument("-batch", "--Batch_size", default=22, help = "Batch Size")

args = parser.parse_args()

img_save_folder_loc	= "generated_images"
TrainBatchSize		= int(args.Batch_size)
device 				= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
loss_fn 			= lpips.LPIPS(net='alex').to(device)


if str(args.mask) == 'thick':
    TrainDataLoaderConfig={'indir': 'celebhq/train_256',
                        'out_size': 256, 
                        'mask_gen_kwargs': 
                                            {'irregular_proba': 1,
                                            'irregular_kwargs': 
                                                                    {'max_angle': 4,
                                                                    'max_len': 200, 
                                                                    'max_width': 100, 
                                                                    'max_times': 5,  
                                                                    'min_times': 1}, 
                                            'box_proba': 0.3, 
                                            'box_kwargs': 
                                                            {'margin': 10,
                                                            'bbox_min_size': 30,
                                                            'bbox_max_size': 150, 
                                                            'max_times': 3, 
                                                            'min_times': 1},
                                            'segm_proba': 0},
                        'transform_variant': 'no_augs', 
                        'dataloader_kwargs': 
                                                {'batch_size': TrainBatchSize, 
                                                 'shuffle': True, 
                                                 'num_workers': 4}
                        } 
	
else:
	TrainDataLoaderConfig={'indir': 'celebhq/train_256',
                        'out_size': 256, 
                        'mask_gen_kwargs': 
                                            {'irregular_proba': 1,
                                            'irregular_kwargs': 
                                                                    {'max_angle': 4,
                                                                    'max_len': 100, 
                                                                    'max_width': 50, 
                                                                    'max_times': 5,  
                                                                    'min_times': 4}, 
                                            'box_proba': 0.3, 
                                            'box_kwargs': 
                                                            {'margin': 0,
                                                            'bbox_min_size': 10,
                                                            'bbox_max_size': 50, 
                                                            'max_times': 5, 
                                                            'min_times': 1},
                                            'segm_proba': 0},
                        'transform_variant': 'no_augs', 
                        'dataloader_kwargs': 
                                                {'batch_size': TrainBatchSize, 
                                                 'shuffle': True, 
                                                 'num_workers': 4}
                        } 


train_loader = make_default_train_dataloader(**TrainDataLoaderConfig)

ValDataLoaderConfig = {	'dataloader_kwargs': {	'batch_size'	: int(TrainBatchSize/2), 
						  						'shuffle'		: False, 
												'num_workers'	: 4}}

eval_loader	= make_default_val_dataloader( indir	= "celebhq/val_256/random_medium_256/",
					  					img_suffix	= ".png", 
										out_size	= 256,
										**ValDataLoaderConfig) 


print(len(train_loader)*int(TrainBatchSize))
print(len(eval_loader)*int(TrainBatchSize/2))


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

summary(model, input_size= [(1, 3, 256,256),(1, 1,256,256)], col_names= ("input_size","output_size","num_params"), depth = 6)



def calc_curr_performance(model,valloader, epoch = 0, full_dataset=True):
	
	Losses = {"L1":[], "L2":[], "PSNR":[], "SSIM":[], "LPIPS":[]}
	
	for i, data in enumerate(tqdm(valloader)):

		if not full_dataset and i>2:
			break

		img, mask 		= torch.Tensor(data["image"].to(device)), torch.Tensor(data["mask"].to(device))
		ground_truth 	= img.clone()
		
		img[:, :, :]	= img[:, :, :] * (1 - mask)
		masked_img	  	= img

		out			 	= model(masked_img, mask)
  
		losses		  	= EvalMetrics(out, ground_truth)

		for metric in losses.keys():
			Losses[metric].append(losses[metric])
	for j in range(3):		
		cv2.imwrite(img_save_folder_loc+"/"+"eval_input"+str(j)+".png",cv2.cvtColor(masked_img[j].permute([1,2,0]).cpu().detach().numpy()*255, cv2.COLOR_RGB2BGR))
		cv2.imwrite(img_save_folder_loc+"/"+"eval_output"+str(j)+".png",cv2.cvtColor(out[j].permute([1,2,0]).cpu().detach().numpy()*255, cv2.COLOR_RGB2BGR))

	return Losses

def EvalMetrics(out,gt):
	losses={}
	
	losses["L1"]	=  nn.L1Loss()(out,gt).mean().item()
	losses["L2"]	=  nn.MSELoss()(out,gt).mean().item()
	losses["PSNR"]  =  peak_signal_noise_ratio(out,gt).mean().item()
	losses["SSIM"]  =  structural_similarity_index_measure(out,gt).mean().item()
	losses["LPIPS"] =  loss_fn(gt,out).mean().item()

	return losses

class HybridLoss(nn.Module):
	def __init__(self, alpha = 0.5):
		super(HybridLoss, self).__init__()
		self.alpha=alpha
		
	def forward(self, x, y):
	
		l_lpips = loss_fn(x, y).mean()
		
		losses = l_lpips + (1 - self.alpha)*(nn.L1Loss()(x, y)) + (self.alpha*(nn.MSELoss()(x, y)))

		return losses*10

scaler 		= torch.cuda.amp.GradScaler()
optimizer 	= optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
criterion 	= HybridLoss().to(device)


prev_loss	 = float("inf")
Losses 		 = calc_curr_performance(model,eval_loader)
Final_losses = {}
for metric in Losses.keys():
	Final_losses[metric] = round(np.array(Losses[metric]).mean(), 4)

print("="*100)
print("####PRE TRAIN:",Final_losses)
print("="*100)

epoch 	= 0
counter = 0
while counter < 10:
	index 			= 0
	epoch_loss  	= 0
	

	model.train()

	with tqdm(train_loader, unit="batch") as tepoch:
		tepoch.set_description(f"Epoch {epoch+1}")

		for i, data in enumerate(tepoch, 0):

			image, mask = data["image"].to(device), data["mask"].to(device)
			target		= image.clone() ## expected output

			image[:, :, :] 	= image[:, :, :] * (1 - mask)
			inputs 			= image
			
			optimizer.zero_grad()
			outputs = model(inputs, mask)
			

			# cv2.imwrite(img_save_folder_loc+"/test_in.png",cv2.cvtColor(inputs[0].permute([1,2,0]).cpu().detach().numpy()*255, cv2.COLOR_RGB2BGR))
			# cv2.imwrite(img_save_folder_loc+"/test_out.png",cv2.cvtColor(outputs[0].permute([1,2,0]).cpu().detach().numpy()*255, cv2.COLOR_RGB2BGR))
			
			with torch.cuda.amp.autocast():
				loss = criterion(outputs, target)
			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()


			epoch_loss += loss.detach() 

			tepoch.set_postfix_str(f" loss : {epoch_loss/len(train_loader):.4f}")

		print(f"Epoch : {epoch+1} - epoch_loss: {epoch_loss}")
			
	index = 0

	for i in range(3):
		cv2.imwrite(img_save_folder_loc+"/"+str(epoch)+"__"+str(i)+"_Input.png",cv2.cvtColor(inputs[index + i].permute([1,2,0]).cpu().detach().numpy()*255, cv2.COLOR_RGB2BGR))
		cv2.imwrite(img_save_folder_loc+"/"+str(epoch)+"__"+str(i)+"_Output.png",cv2.cvtColor(outputs[index + i].permute([1,2,0]).cpu().detach().numpy()*255, cv2.COLOR_RGB2BGR))
	
	counter += 1
	model.eval()
	if epoch % 1 == 0:

		Losses = calc_curr_performance(model,eval_loader,epoch)
		Final_losses = {}
		for metric in Losses.keys():
			Final_losses[metric] = round(np.array(Losses[metric]).mean(), 4)

		print("####epoch ",epoch+1,"Testing: ",Final_losses)

		if prev_loss >= Final_losses["LPIPS"]:
			torch.save(model.state_dict(), PATH)
			prev_loss = Final_losses["LPIPS"]
			print("saving chkpoint")
			counter = 0

	
	epoch += 1

print("Switching to SGD")

model.load_state_dict(torch.load(PATH))
optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)
counter = 0
while counter < 10:
	index 		= 0
	epoch_loss  = 0
	
	model.train()

	with tqdm(train_loader, unit="batch") as tepoch:
		tepoch.set_description(f"Epoch {epoch+1}")

		for i, data in enumerate(tepoch, 0):

			image, mask = data["image"].to(device), data["mask"].to(device)
			target		= image.clone() ## expected output

			image[:, :, :] 	= image[:, :, :] * (1 - mask)
			inputs 			= image
			
			optimizer.zero_grad()
			
			outputs = model(inputs, mask)

			# cv2.imwrite("img_save_folder/"+img_save_folder_loc+"/test_in.png",cv2.cvtColor(inputs[0].permute([1,2,0]).cpu().detach().numpy()*255, cv2.COLOR_RGB2BGR))
			# cv2.imwrite("img_save_folder/"+img_save_folder_loc+"/test_out.png",cv2.cvtColor(outputs[0].permute([1,2,0]).cpu().detach().numpy()*255, cv2.COLOR_RGB2BGR))
			
			with torch.cuda.amp.autocast():
				loss = criterion(outputs, target)
			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()


			epoch_loss += loss.detach() 

			tepoch.set_postfix_str(f" loss : {epoch_loss/len(train_loader):.4f}" )

		print(f"Epoch : {epoch+1} - epoch_loss: {epoch_loss}" )
			
	index = 0
	for i in range(3):
		cv2.imwrite(img_save_folder_loc+"/"+str(epoch)+"__"+str(i)+"_Input.png",cv2.cvtColor(inputs[index + i].permute([1,2,0]).cpu().detach().numpy()*255, cv2.COLOR_RGB2BGR))
		cv2.imwrite(img_save_folder_loc+"/"+str(epoch)+"__"+str(i)+"_Output.png",cv2.cvtColor(outputs[index + i].permute([1,2,0]).cpu().detach().numpy()*255, cv2.COLOR_RGB2BGR))
	
	counter += 1
	model.eval()
	if epoch % 1 == 0:

		Losses = calc_curr_performance(model,eval_loader,epoch)
		Final_losses = {}
		for metric in Losses.keys():
			Final_losses[metric] = round(np.array(Losses[metric]).mean(), 4)

		print("####epoch ",epoch+1,"Testing: ",Final_losses)

		if prev_loss >= Final_losses["LPIPS"]:
			torch.save(model.state_dict(), PATH)
			prev_loss = Final_losses["LPIPS"]
			print("saving chkpoint")
			counter = 0

	
	epoch += 1

print('Finished Training')

model.load_state_dict(torch.load(PATH))
Losses 		 = calc_curr_performance(model, eval_loader)
Final_losses = {}
for metric in Losses.keys():
	Final_losses[metric] = round(np.array(Losses[metric]).mean(), 4)

print("="*100)
print("####BEST MODEL:",Final_losses)
print("="*100)


