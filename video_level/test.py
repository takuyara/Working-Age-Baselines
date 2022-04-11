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
from scipy.stats import pearsonr

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
	acc = np.mean(yp == yt)
	"""
	pcc, __ = pearsonr(yt, yp)
	theta_t, theta_p = np.var(yt), np.var(yp)
	ccc = (2 * pcc * theta_p * theta_t) / (theta_p ** 2 + theta_t ** 2 + (np.mean(yt) - np.mean(yp)) ** 2)
	"""
	return acc
	
def main():
	args = get_args()
	test_ids = get_partitions(args.feature_path, args.cur_fold)["val"]
	if args.integrator == "SPECTRAL":
		transform = SpectralRepr(args.integrate_length, args.hidden_param)
		integrator = None
		in_rate = 2
		min_len = args.hidden_param
	else:
		def get_mid(x, l):
			st = (x.shape[0] - l) // 2
			return x[st : st + l, ...]
		transform = get_mid
		integrator = LSTMInteg(args.integrate_length, args.hidden_param)
		in_rate = 1
		min_len = args.integrate_length
	datasets = {task : VideoDataset(args.feature_path, args.label_path, test_ids, args.predict_dim, min_len, transform = transform, required_task = task) for task in config.all_tasks}
	resume_data = torch.load(args.resume_path)
	if "config" not in resume_data:
		resume_data = {"model": resume_data, "config": args.model_config}
	device = torch.device(args.device)
	model = ModelWrapper(integrator, CNNClassifier(resume_data["config"], datasets[config.all_tasks[0]].feature_shape * in_rate, num_classes))
	model.load_state_dict(resume_data["model"])
	model = model.to(device)
	model.eval()
	print(args.resume_path)
	for task in config.all_tasks:
		acc = solve_single(model, DataLoader(datasets[task], batch_size = args.batch_size), device)
		print(acc)

if __name__ == '__main__':
	main()
