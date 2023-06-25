
import torch, torchvision
import torch.nn as nn
import numpy as np
import cv2, time, lpips, argparse 
from tqdm import tqdm
from torchinfo import summary
from torchmetrics.functional import peak_signal_noise_ratio
from torchmetrics.functional import structural_similarity_index_measure
from saicinpainting.training.data.datasets import make_default_train_dataloader, make_default_val_dataloader
from model import WavePaint
import torch.optim as optim



parser = argparse.ArgumentParser()


parser.add_argument("-vis", "--Visual_example", default="Train_samples", help = "Path to save visual examples")
parser.add_argument("-batch", "--Batch_size", default=216, help = "Batch Size")

args = parser.parse_args()

Visual_example_loc	= str(args.Visual_example)
TrainBatchSize		= int(args.Batch_size)
device 				= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
loss_fn 			= lpips.LPIPS(net='alex').to(device)




TrainDataLoaderConfig = 	{	'indir'				: 	'ImageNet/train',
								'out_size'			:	 224,
								'mask_gen_kwargs'	: { 	'irregular_proba'	: 1,
															'irregular_kwargs'	: {	'max_angle'		: 4, 
																					'max_len'		: 35,
																					'max_width'		: 30,
																					'max_times'		: 10,
																					'min_times'		: 4},
															'box_proba'			: 1, 
															'box_kwargs'		: {	'margin'		: 0,
																					'bbox_min_size'	: 30,
																					'bbox_max_size'	: 75, 
																					'max_times'		: 5,
																					'min_times'		: 2},
															'segm_proba'			: 0},
								'transform_variant'	: 		'distortions', 
								'dataloader_kwargs'	: {		'batch_size'		: TrainBatchSize,
															'shuffle'			: True,
															'num_workers'		: 4}}  ### IMAGENET

train_loader = make_default_train_dataloader(**TrainDataLoaderConfig)

ValDataLoaderConfig = {	'dataloader_kwargs': {	'batch_size': int(TrainBatchSize/2), 
						  						'shuffle'	: False, 
												'num_workers': 4}}

eval_loader	=	make_default_val_dataloader(	indir		= "ImageNet/valfolder/random_medium_224/",
					  							img_suffix	= ".png", 
												out_size	= 224,
												 **ValDataLoaderConfig) # val loader


print(len(train_loader)*int(TrainBatchSize))
print(len(eval_loader)*int(TrainBatchSize/2))


### MODEL ARCHITECTURE ###


MASK_SIZE=12
REDUCTION_CONV=1
NUM_DWT=1 
MAX_EVAL=10
base_path="WavePaint"
extra="_thin_mask"


NUM_MODELS 		= 	2
MODEL_DEPTH		=	4 
FF_CHANNEL		=	128
MODEL_EMBEDDING	=	128 


model = WavePaint(
	num_modules			= NUM_MODELS,
	blocks_per_module 	= MODEL_DEPTH,
	mult 				= 2,
	ff_channel 			= FF_CHANNEL,
	final_dim 			= MODEL_EMBEDDING,
	dropout 			= 0.5
).to(device)

# PATH = base_path + str('chkpoint__IMG__'+str(MASK_SIZE)+'__MODEL__D'+str(MODEL_DEPTH)+'_E'+str(MODEL_EMBEDDING)+'_N'+str(NUM_MODELS)+'_C'+str(REDUCTION_CONV)+'_F'+str(FF_CHANNEL)+'_#dwt='+str(NUM_DWT)+extra+'.pth')
# PATH = 'WavemixModelschkpoint__IMG__12__MODEL__D7_E128_N1_C1_F128_#dwt=1_thin_maskLayer_newSSIM.pth'
# model.load_state_dict(torch.load(PATH))
PATH = base_path + '_blocks'+str(MODEL_DEPTH)+'_dim'+str(MODEL_EMBEDDING)+'_modules'+str(NUM_MODELS)+'imagenet_depthconv.pth'




summary(model, input_size= [(1, 3,224,224),(1, 1,224,224)], col_names= ("input_size","output_size","num_params"), depth = 6)
###########################################################################################################################################


def calc_curr_performance(model,valloader, epoch = 0, full_dataset=True):
	
	Losses = {"L1":[], "L2":[], "PSNR":[], "SSIM":[], "LPIPS":[]}
	
	for i, data in enumerate(tqdm(valloader)):

		if not full_dataset and i>2:
			break

		img, mask = torch.Tensor(data["image"].to(device)), torch.Tensor(data["mask"].to(device))
		ground_truth = img.clone()
		
		img[:, :, :]	= img[:, :, :] * (1 - mask)
		masked_img	  = img

		out			 = model(masked_img, mask)
		losses		  = EvalMetrics(out, ground_truth)

		for metric in losses.keys():
			Losses[metric].append(losses[metric])

	cv2.imwrite("Visual_example/"+Visual_example_loc+"/"+"eval_input"+str(epoch)+".png",cv2.cvtColor(masked_img[0].permute([1,2,0]).cpu().detach().numpy()*255, cv2.COLOR_RGB2BGR))
	cv2.imwrite("Visual_example/"+Visual_example_loc+"/"+"eval_output"+str(epoch)+".png",cv2.cvtColor(out[0].permute([1,2,0]).cpu().detach().numpy()*255, cv2.COLOR_RGB2BGR))

	return Losses

def EvalMetrics(out,gt):
	losses={}
	
	
	losses["L1"]	=   nn.L1Loss()(out,gt).mean().item()
	losses["L2"]	=   nn.MSELoss()(out,gt).mean().item()
	losses["PSNR"]  =   peak_signal_noise_ratio(out,gt).mean().item()
	losses["SSIM"]  =   structural_similarity_index_measure(out,gt).mean().item()
	losses["LPIPS"] =   loss_fn(gt,out).mean().item()

	return losses

class HybridLoss(nn.Module):
	def __init__(self, alpha=0.5):
		super(HybridLoss, self).__init__()
		self.alpha=alpha
		
	def forward(self, x, y):
	
		l_lpips = loss_fn(x, y).mean()
		
		losses = l_lpips + (1 - self.alpha)*(nn.L1Loss()(x, y )) + (self.alpha*(nn.MSELoss()(x, y )))

		return losses*10

scaler 		= 	torch.cuda.amp.GradScaler()
optimizer 	= 	optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
criterion 	= 	HybridLoss().to(device)


prev_loss	=	float("inf")
Losses 		=	calc_curr_performance(model,eval_loader)
Final_losses={}
for metric in Losses.keys():
	Final_losses[metric] = round(np.array(Losses[metric]).mean(), 4)
print("="*100)
print("####PRE TRAIN:",Final_losses)
print("="*100)


epoch 	= 0
counter = 0
while counter < 5:

	index 			= 0
	running_loss	= 0
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

			# cv2.imwrite("Visual_example/"+Visual_example_loc+"/test_in.png",cv2.cvtColor(inputs[0].permute([1,2,0]).cpu().detach().numpy()*255, cv2.COLOR_RGB2BGR))
			# cv2.imwrite("Visual_example/"+Visual_example_loc+"/test_out.png",cv2.cvtColor(outputs[0].permute([1,2,0]).cpu().detach().numpy()*255, cv2.COLOR_RGB2BGR))
			
			with torch.cuda.amp.autocast():
				loss = criterion(outputs, target)
			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()


			epoch_loss += loss.detach() 

			running_loss += loss.detach() / len(train_loader)
		
			# if i % int(len(train_loader)/500)==int(len(train_loader)/500)-1:	# print every 2000 mini-batches
			if i == int(len(train_loader)):	
				print('[%d, %5d] loss: %.3f' %
					  (epoch + 1, i + 1, epoch_loss))
				running_loss = 0.0
				# break

			tepoch.set_postfix_str(f" loss : {running_loss:.4f}" )

	index = 0
	for i in range(3):
		cv2.imwrite("Visual_example/"+Visual_example_loc+"/"+str(epoch)+"__"+str(i)+"_Input.png",cv2.cvtColor(inputs[index + i].permute([1,2,0]).cpu().detach().numpy()*255, cv2.COLOR_RGB2BGR))
		cv2.imwrite("Visual_example/"+Visual_example_loc+"/"+str(epoch)+"__"+str(i)+"_Output.png",cv2.cvtColor(outputs[index + i].permute([1,2,0]).cpu().detach().numpy()*255, cv2.COLOR_RGB2BGR))
	
	counter += 1
	model.eval()
	if epoch % 1 == 0:

		Losses = calc_curr_performance(model,eval_loader,epoch)
		Final_losses = {}
		for metric in Losses.keys():
			Final_losses[metric]= round(np.array(Losses[metric]).mean(), 4)

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

while counter < 5:

	index 			= 0
	running_loss	= 0
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

			# cv2.imwrite("Visual_example/"+Visual_example_loc+"/test_in.png",cv2.cvtColor(inputs[0].permute([1,2,0]).cpu().detach().numpy()*255, cv2.COLOR_RGB2BGR))
			# cv2.imwrite("Visual_example/"+Visual_example_loc+"/test_out.png",cv2.cvtColor(outputs[0].permute([1,2,0]).cpu().detach().numpy()*255, cv2.COLOR_RGB2BGR))
			
			with torch.cuda.amp.autocast():
				loss = criterion(outputs, target)
			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()


			epoch_loss += loss.detach() 

			running_loss += loss.detach() / len(train_loader)
		
			if i % int(len(train_loader)/500)==int(len(train_loader)/500)-1:	# print every 2000 mini-batches
			# if i == int(len(train_loader)):	
				print('[%d, %5d] loss: %.3f' %
					  (epoch + 1, i + 1, epoch_loss))
				running_loss = 0.0
				break

			tepoch.set_postfix_str(f" loss : {running_loss:.4f}" )

	index = 0
	for i in range(3):
		cv2.imwrite("Visual_example/"+Visual_example_loc+"/"+str(epoch)+"__"+str(i)+"_Input.png",cv2.cvtColor(inputs[index + i].permute([1,2,0]).cpu().detach().numpy()*255, cv2.COLOR_RGB2BGR))
		cv2.imwrite("Visual_example/"+Visual_example_loc+"/"+str(epoch)+"__"+str(i)+"_Output.png",cv2.cvtColor(outputs[index + i].permute([1,2,0]).cpu().detach().numpy()*255, cv2.COLOR_RGB2BGR))
	
	counter += 1
	model.eval()
	if epoch % 1 == 0:

		Losses = calc_curr_performance(model,eval_loader,epoch)
		Final_losses = {}
		for metric in Losses.keys():
			Final_losses[metric]= round(np.array(Losses[metric]).mean(), 4)

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
