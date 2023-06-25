import os
from tqdm import tqdm
import shutil
import numpy as np

source_folder = "/workspace/ImageNet/224Val"

sub_folders=os.listdir(source_folder)
# print(sub_folders[0])

to_folder = "/workspace/Darshan/lama/ImageNet"

options=["valfolder"]


for subfolder in tqdm(sub_folders):
	curr_path = os.path.join(source_folder,subfolder)
	for file in os.listdir(curr_path):
		choice = np.random.choice(options)
		file_path=os.path.join(curr_path,file)
		new_path = os.path.join(to_folder,choice)
		new_path = os.path.join(new_path,file)

		shutil.copy(file_path, new_path)


