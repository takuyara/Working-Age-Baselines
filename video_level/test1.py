# Pionniers du TJ, benissiez-moi par votre Esprits Saints!
import sys
sys.path.append("../")

from dataset import VideoDataset1
from arguments import get_video_parser
from utils import num_classes, get_partitions, get_info
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
	resids = []
	for x, yt, part_id in dataloader:
		# print(part_id, type(part_id))
		x = x.to(device)
		with torch.no_grad():
			y = model(x)
			__, yp = torch.max(y, dim = 1)
			resyp.append(yp.cpu().numpy())
			resids += list(part_id)
	yp = np.concatenate(resyp, axis = 0).tolist()
	"""
	pcc, __ = pearsonr(yt, yp)
	theta_t, theta_p = np.var(yt), np.var(yp)
	ccc = (2 * pcc * theta_p * theta_t) / (theta_p ** 2 + theta_t ** 2 + (np.mean(yt) - np.mean(yp)) ** 2)
	"""
	return zip(resids, yp)
	
def main(args):
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
	datasets = {task : VideoDataset1(args.feature_path, args.label_path, test_ids, args.predict_dim, min_lens[args.integrator], select_interval = args.select_interval, transform = transforms[args.integrator], required_task = task, cache_transform = args.cache_transform) for task in config.all_tasks}
	dataloaders = {task : DataLoader(t_dataset, batch_size = args.batch_size) for task, t_dataset in datasets.items()}
	feature_shape = datasets[config.all_tasks[0]].feature_shape
	in_channels = {"SPECTRAL": 2 * feature_shape, "LSTM": 2 * args.hidden_param, "MODE": 1, "AVERAGE": feature_shape}
	integrators = {"SPECTRAL": None, "LSTM": LSTMInteg(feature_shape, args.hidden_param), "MODE": None, "AVERAGE": None}
	resume_data = torch.load(args.resume_path)
	if "config" not in resume_data:
		resume_data = {"model": resume_data, "config": args.model_config}
	#print(resume_data["config"])
	device = torch.device(args.device)
	model = ModelWrapper(integrators[args.integrator], classifiers[args.integrator](resume_data["config"], in_channels[args.integrator], num_classes))
	model.load_state_dict(resume_data["model"])
	model = model.to(device)
	model.eval()
	res = []
	for task in config.all_tasks:
		task_res = solve_single(model, dataloaders[task], device)
		for part_id, y_pred in task_res:
			res.append(get_info(part_id) + [y_pred])
	with open("./predictions.csv", "a", newline = "") as f:
		writer = csv.writer(f)
		writer.writerows(res)


if __name__ == '__main__':
	args = get_args()
	main(args)
