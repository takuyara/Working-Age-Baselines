# Pionniers du TJ, benissiez-moi par votre Esprits Saints!
import sys
sys.path.append("../")

from dataset import VideoDataset
from arguments import get_video_parser
from utils import num_classes, get_partitions
import os
import csv
import numpy as np
import torch
import config
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from model import *
from integrator import *
from model_wrapper import ModelWrapper
from scipy import stats

logger = logging.getLogger(__name__)

def get_args():
	parser = get_video_parser()
	parser.add_argument("-rp", "--resume-path", type = str, default = None)
	args = parser.parse_args()
	path = args.model_config
	if os.path.exists(path):
		r_config = []
		with open(path, newline = "") as f:
			reader = csv.DictReader(f)
			for row in reader:
				r_config.append((int(row["kernels"]), int(row["channels"]), float(row["dropout"])))
		args.model_config = r_config
	else:
		args.model_config = None
	return args

def solve_single(model, dataloader, device):
	resyp = []
	resyt = []
	for x, yt in dataloader:
		x = x.to(device)
		with torch.no_grad():
			y = model(x)
			__, yp = torch.max(y, dim = 1)
			resyp.append(yp.cpu().numpy())
			resyt.append(yt.cpu().numpy())
	yp = np.concatenate(resyp, axis = 0)
	yt = np.concatenate(resyt, axis = 0)
	uars = {}
	for i in yp.shape[0]:
		uars.setdefault(yt[i], []).append(1 if yp[i] == yt[i] else 0)
	return uars
	
def infer_partial(args):
	test_ids = get_partitions(args.feature_path, args.cur_fold, args.gender)["val"]
	def get_mid(x):
		st = (x.shape[0] - args.integrate_length) // 2
		return x[st : st + args.integrate_length, ...]
	def mode_onehot(x):
		t, __ = stats.mode(x)
		t = np.eye(num_classes)[t.astype("uint8")].flatten()
		return t
	min_lens = {"SPECTRAL": args.hidden_param, "LSTM": args.integrate_length, "MODE": 0, "AVERAGE": 0}
	transforms = {"SPECTRAL": SpectralRepr(args.integrate_length, args.hidden_param), "LSTM": get_mid, "MODE": mode_onehot, "AVERAGE": lambda x : np.mean(x, axis = 0)}
	classifiers = {"SPECTRAL": CNNClassifier, "LSTM": CNNClassifier, "MODE": NullClassifier, "AVERAGE": MLPClassifier}
	datasets = {task : VideoDataset(args.feature_path, args.label_path, test_ids, args.predict_dim, min_lens[args.integrator], select_interval = args.select_interval, transform = transforms[args.integrator], required_task = task, cache_transform = args.cache_transform) for task in config.all_tasks}
	dataloaders = {task : DataLoader(t_dataset, batch_size = args.batch_size) for task, t_dataset in datasets.items()}
	feature_shape = datasets[config.all_tasks[0]].feature_shape
	in_channels = {"SPECTRAL": 2 * feature_shape, "LSTM": 2 * args.hidden_param, "MODE": 1, "AVERAGE": feature_shape}
	integrators = {"SPECTRAL": None, "LSTM": LSTMInteg(feature_shape, args.hidden_param), "MODE": None, "AVERAGE": None}
	resume_data = torch.load(args.resume_path)
	if "config" not in resume_data:
		resume_data = {"model": resume_data, "config": args.model_config}
	device = torch.device(args.device)
	model = ModelWrapper(integrators[args.integrator], classifiers[args.integrator](resume_data["config"], in_channels[args.integrator], num_classes))
	model.load_state_dict(resume_data["model"])
	model = model.to(device)
	model.eval()
	all_uars = {}
	task_uar = {}
	for task in config.all_tasks:
		this_task_uar = solve_single(model, dataloaders[task], device)
		task_uar[task] = this_task_uar
		for yt, accs in this_task_uar.items():
			all_uars.setdefault(yt, []).extend(accs)
	task_uar["Total"] = all_uars
	return task_uar

def main():
	# base_path = "/rds/user/yl847/hpc-work/outlast-f/"
	base_path = "D:\\working-age-data"
	args = get_args()
	total_result = []
	for predict_dim, predict_name in enumerate(["v", "a"]):
		for input_feature_type in ["GraphAU-P", "GraphAU-F", "ResNet-P", "ResNet-F"]:
			for long_term_strategy in ["SPECTRAL", "LSTM", "MODE", "AVERAGE"]:
				dim_uars = {}
				for site in range(4):				
					args.hidden_param = 16 if long_term_strategy == "LSTM" and input_feature_type.endswith("P") else 256
					args.integrator = long_term_strategy
					args.cur_fold = site
					args.predict_dim = predict_dim
					if input_feature_type.endswith("P"):
						args.model_config = "model_config_p.csv"
					elif input_feature_type.startswith("GraphAU"):
						args.model_config = "model_config_f_au.csv"
					else:
						args.model_config = "model_config_f_resnet.csv"
					args.label_path = os.path.join(base_path, "questionnaire")
					args.feature_path = os.path.join(base_path, f"{input_feature_type}-{site}-{predict_name}")
					res_uars = infer_partial(args)
					for task, uars in res_uars.items():
						for yt, accs in uars.items():
							dim_uars.setdefault(task, {}).setdefault(yt, []).extend(accs)
				for task, uars in dim_uars.items():
					average_accs = []
					for yt, accs in uars.items():
						average_accs.append(sum(accs) / len(accs))
						total_result.append((predict_name, input_feature_type, long_term_strategy, task, sum(average_accs) / len(average_accs)))
	with open("all_result.csv", newline = "") as f:
		writer = csv.writer(f)
		f.writerows(total_result)
if __name__ == '__main__':
	args = get_args()
	main(args)
