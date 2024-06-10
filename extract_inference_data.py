import argparse, logging, os, torch, ee_dnns, config
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import pandas as pd


def extracting_ee_inference_data(args, test_loader, ee_model, device, threshold):

	n_exits = args.n_branches + 1	
	conf_list, correct_list, delta_inf_time_list, cum_inf_time_list = [], [], [], []
	prediction_list, target_list = [], []

	model.eval()
	with torch.no_grad():
		for (data, target) in tqdm(test_loader):	

			# Convert data and target into the current device.
			data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

			# Obtain confs and predictions for each side branch.
			conf_branches, predictions_branches, delta_inf_time_branches, cum_inf_time_branches = model.forwardEval(data)

			conf_list.append(conf_branches), delta_inf_time_list.append(delta_inf_time_branches), cum_inf_time_list.append(cum_inf_time_branches)

			correct_list.append([predictions_branches[i].eq(target.view_as(predictions_branches[i])).sum().item() for i in range(n_exits)])
			target_list.append(target.item()), prediction_list.append(predictions_branches)

	conf_list, correct_list, delta_inf_time_list = np.array(conf_list), np.array(correct_list), np.array(delta_inf_time_list)
	cum_inf_time_list, prediction_list = np.array(cum_inf_time_list), np.array(prediction_list)

	accuracy_branches = [sum(correct_list[:, i])/len(correct_list[:, i]) for i in range(n_exits)]

	print("Accuracy: %s"%(accuracy_branches))
	
	result_dict = {"threshold": len(target_list)*[threshold], "device": len(target_list)*[str(device)], "target": target_list}

	for i in range(n_exits):
		result_dict["conf_branch_%s"%(i+1)] = conf_list[:, i]
		result_dict["correct_branch_%s"%(i+1)] = correct_list[:, i]
		result_dict["delta_inf_time_branch_%s"%(i+1)] = delta_inf_time_list[:, i]
		result_dict["cum_inf_time_branch_%s"%(i+1)] = cum_inf_time_list[:, i]
		result_dict["prediction_branch_%s"%(i+1)] = prediction_list[:, i]

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

	device = torch.device('cuda' if (torch.cuda.is_available() and args.use_gpu) else 'cpu')

	model_path = os.path.join(DIR_PATH, "models", "%s_model_%s_full.pt"%(args.model_name, args.loss_weights_type))

	#indices_path = os.path.join(config.DIR_PATH, "indices_%s.pt"%(args.dataset_name))

	inf_data_dir_path = os.path.join(DIR_PATH, "inf_data_results")
	os.makedirs(inf_data_dir_path, exist_ok=True)

	inf_data_path = os.path.join(inf_data_dir_path, "inf_data_ee_%s_%s.csv"%(args.model_name, args.loss_weights_type))
		
	test_data = riq.IQDataset(data_folder=data_folder, chunk_size=chunk_size, stride=sample_stride, subset='test')
	test_data.normalize(torch.tensor([-2.7671e-06, -7.3102e-07]), torch.tensor([0.0002, 0.0002]))
	test_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False, num_workers=loaders, pin_memory=True)

	ee_model = load_eednn_model(args, n_classes, model_path, device)

	threshold_list = np.linspace(0.6, 0.95, 8)
	threshold_list = np.append(threshold_list, 1)

	for threshold in threshold_list:
		print("Threshold: %s"%(threshold))

		df_inf_data = extracting_ee_inference_data(args, test_loader, ee_model, device, threshold)

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

	parser.add_argument('--distribution', type=str, default=config.distribution, 
		help='Distribution of the early exits. Default: predefined')

	parser.add_argument('--model_id', type=int, help='Model_id.')

	parser.add_argument('--loss_weights_type', default="decrescent", type=str, help='loss_weights_type.')


	args = parser.parse_args()

	main(args)