import argparse, logging, os, torch, ee_dnns, config
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import pandas as pd
import readCharmDataset as riq
from pthflops import count_ops
import ptflops

def extracting_ee_inference_data(args, test_loader, model, device):

	n_exits = args.n_branches + 1	
	flops_list = []

	model.eval()
	with torch.no_grad():
		for (data, target) in tqdm(test_loader):	

			# Convert data and target into the current device.
			data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

			# Obtain confs and predictions for each side branch.

			flops = model.forwardFlops(data)
			#print(flops)
			#sys.exit()
			flops_list.append(flops)
			break

	
	flops_list = np.array(flops_list)

	
	result_dict = {}

	#print("Flops: %s"%(flops_list.mean()))

	for i in range(n_exits):
		result_dict["flops_branch_%s"%(i+1)] = flops_list[:, i]


	#Converts to a DataFrame Format.
	df = pd.DataFrame(np.array(list(result_dict.values())).T, columns=list(result_dict.keys()))

	# Returns confidences and predictions into a DataFrame.
	return df

def load_eednn_model(args, n_classes, model_path, device):

	#Instantiate the Early-exit DNN model.
	ee_model = ee_dnns.Early_Exit_DNN(args.model_name, 3, args.n_branches, args.exit_type, device, exit_positions=config.exit_positions)    

	#Load the trained early-exit DNN model.
	ee_model.load_state_dict(torch.load(model_path, map_location=device)["model_state_dict"])
	ee_model = ee_model.to(device)

	return ee_model

def main(args):

	#n_classes = config.n_class_dict[args.dataset_name]
	DIR_PATH = os.path.dirname(__file__)

	device = torch.device('cuda')

	model_path = os.path.join(DIR_PATH, "models", "%s_model_%s_full.pt"%(args.model_name, args.loss_weights_type))

	#indices_path = os.path.join(config.DIR_PATH, "indices_%s.pt"%(args.dataset_name))

	inf_data_dir_path = os.path.join(DIR_PATH, "inf_data_results")
	os.makedirs(inf_data_dir_path, exist_ok=True)

	inf_data_path = os.path.join(inf_data_dir_path, "flops_inf_data_ee_%s_%s_final.csv"%(args.model_name, args.loss_weights_type))
		
	test_data = riq.IQDataset(data_folder="./oran_dataset", chunk_size=20000, stride=0, subset='validation')
	test_data.normalize(torch.tensor([-2.7671e-06, -7.3102e-07]), torch.tensor([0.0002, 0.0002]))
	test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=6, pin_memory=True)

	ee_model = load_eednn_model(args, 3, model_path, device)

	df_inf_data = extracting_ee_inference_data(args, test_loader, ee_model, device)

	df_inf_data.to_csv(inf_data_path, mode='a', header=not os.path.exists(inf_data_path))




if (__name__ == "__main__"):
	# Input Arguments to configure the early-exit model .
	parser = argparse.ArgumentParser(description="Extract the confidences obtained by DNN inference for next experiments.")

	#We here insert the argument dataset_name. 
	#The initial idea is this novel calibration method evaluates three dataset for image classification: cifar10, cifar100 and
	#caltech256. First, we implement caltech256 dataset.
	parser.add_argument('--model_name', type=str, default="cnn", 
		choices=["cnn", "rn"], help='Model name.')

	#We here insert the argument model_name. 
	#We evalue our novel calibration method Offloading-driven Temperature Scaling in four early-exit DNN:
	#MobileNet

	parser.add_argument('--seed', type=int, default=42, help='Seed.')

	parser.add_argument('--use_gpu', type=bool, default=True, help='Use GPU? Default: True')

	parser.add_argument('--n_branches', type=int, default=2, help='Number of side branches.')

	parser.add_argument('--exit_type', type=str, default="bnpool", 
		help='Exit Type. Default: bnpool')

	parser.add_argument('--distribution', type=str, default="predefined", 
		help='Distribution of the early exits. Default: predefined')

	parser.add_argument('--model_id', type=int, help='Model_id.')

	parser.add_argument('--loss_weights_type', default="decrescent", type=str, help='loss_weights_type.')


	args = parser.parse_args()

	main(args)