

NUM_MODULES 	= 8
NUM_BLOCKS	= 4 
MODEL_EMBEDDING	= 128 


model = WavePaint(
	num_modules			    = 8,
	blocks_per_module 	= 4,
	mult 				        = 4,
	ff_channel 			    = 128,
	final_dim 			    = 128,
	dropout 			      = 0.5
).to(device)
