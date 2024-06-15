import argparse, logging, os, torch, config, cnn_model, sys
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import pandas as pd
import readCharmDataset as riq


def extracting_inf_time(args, test_loader, model, device):

	flops_list = []

	model.eval()
	with torch.no_grad():
		for (data, target) in tqdm(test_loader):	

			# Convert data and target into the current device.
			data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

			# Obtain confs and predictions for each side branch.
			flop, _ = count_ops(model, data, print_readable=False, verbose=False)

			flops_list.append(flop)

	flops_list = np.array(flops_list)

	result_dict = {"device": len(flops_list)*[str(device)], "flops": flops_list}

	#Converts to a DataFrame Format.
	df = pd.DataFrame(np.array(list(result_dict.values())).T, columns=list(result_dict.keys()))

	# Returns confidences and predictions into a DataFrame.
	return df


def main(args):

	DIR_PATH = os.path.dirname(__file__)

	device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')

	inf_data_dir_path = os.path.join(DIR_PATH, "inf_data_results")
	os.makedirs(inf_data_dir_path, exist_ok=True)

	flops_path = os.path.join(inf_data_dir_path, "flops_dnn_inf_data_%s.csv"%(args.model_name))

	test_data = riq.IQDataset(data_folder="./oran_dataset", chunk_size=20000, stride=0, subset='test')
	test_data.normalize(torch.tensor([-2.7671e-06, -7.3102e-07]), torch.tensor([0.0002, 0.0002]))
	test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=6, pin_memory=True)

	model = cnn_model.ConvModel().to(device)

	df_flops = extracting_flops(args, test_loader, model, device)

	df_flops.to_csv(flops_path, mode='a', header=not os.path.exists(flops_path))




if (__name__ == "__main__"):
	# Input Arguments to configure the early-exit model .
	parser = argparse.ArgumentParser(description="Extract the confidences obtained by DNN inference for next experiments.")

	#We here insert the argument dataset_name. 
	#The initial idea is this novel calibration method evaluates three dataset for image classification: cifar10, cifar100 and
	#caltech256. First, we implement caltech256 dataset.
	parser.add_argument('--model_name', type=str, default="cnn", 
		choices=["cnn", "rn"], help='Model name.')

	parser.add_argument('--use_gpu', type=bool, default=False, help='Use GPU? Default: True')

	args = parser.parse_args()

	main(args)