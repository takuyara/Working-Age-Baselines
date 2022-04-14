# Pionniers du TJ, benissiez-moi par votre Esprits Saints!
import sys
sys.path.append("../")

import os
import csv
import time
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import Compose
from tqdm import tqdm
from copy import deepcopy
from scipy import stats
import logging
logger = logging.getLogger(__name__)

from model import *
from integrator import *
from model_wrapper import ModelWrapper
from dataset import VideoDataset
from utils import get_partitions, num_classes, get_identifier, get_dim_name
from arguments import get_video_parser
from test import main as test

def get_args():
	parser = get_video_parser()
	args = parser.parse_args()
	path = args.model_config
	r_config = []
	with open(path, newline = "") as f:
		reader = csv.DictReader(f)
		for row in reader:
			r_config.append((int(row["kernels"]), int(row["channels"]), float(row["dropout"])))
	args.model_config = r_config
	return args

def get_mean(x):
	return sum(x) / len(x)

def train_val(model, dataloaders, criterion, optimizer, epochs, patience, device, print_interval, consistent_model):
	min_loss, max_acc = 1e100, 0
	stop_training = False
	for i in range(1, epochs + 1):
		print("Epoch ", i)
		sttime = time.time()
		for phase in ["train", "val"]:
			e_loss = []
			e_acc = []
			if phase == "train":
				model.train()
			else:
				model.eval()
			if print_interval is None:
				pbar = tqdm(dataloaders[phase])
				pbar.set_description(phase)
			else:
				pbar = dataloaders[phase]
				print(f"{phase.capitalize()}:")
			for j, (x, y) in enumerate(pbar):
				optimizer.zero_grad()
				x, y = x.to(device), y.to(device)
				with torch.set_grad_enabled(phase == "train"):
					yp = model(x)
					b_loss = criterion(yp, y)
					if phase == "train" and not consistent_model:
						b_loss.backward()
						optimizer.step()
					__, lp = torch.max(yp, dim = 1)
					b_acc = torch.sum(lp == y) / y.shape[0]
					b_acc = b_acc.item()
					b_loss = b_loss.item()
					if print_interval is None:
						pbar.set_postfix({"loss": f"{b_loss:.4f}", "acc": f"{b_acc * 100:.2f}"})
					e_loss.append(b_loss)
					e_acc.append(b_acc)
				if (print_interval is not None and (j + 1) % print_interval == 0) or j == len(pbar) - 1:
					print(f"[{j + 1}/{len(pbar)}]: loss = {get_mean(e_loss):.4f}, acc = {get_mean(e_acc) * 100:.2f}", flush = True)
			e_loss = sum(e_loss) / len(e_loss)
			e_acc = sum(e_acc) / len(e_acc)
			if phase == "val":
				if e_loss < min_loss:
					min_loss, min_epoch = e_loss, i
				elif i - min_epoch >= patience:
					stop_training = True
				if e_acc > max_acc:
					max_acc = e_acc
					print("new_max_acc ", e_acc)
					best_state = deepcopy(model.state_dict())
		# print(f"Epoch time: {time.time() - sttime}s")
		if stop_training:
			break
	return max_acc, best_state

def main():
	args = get_args()
	print(args.feature_path)
	partitions = get_partitions(args.feature_path, args.cur_fold)
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
	datasets = {phase : VideoDataset(args.feature_path, args.label_path, partitions[phase], args.predict_dim, min_lens[args.integrator], transform = transforms[args.integrator], required_task = None, cache_transform = args.cache_transform) for phase in ["train", "val"]}
	dataloaders = {phase : DataLoader(t_dataset, batch_size = args.batch_size, shuffle = True) for phase, t_dataset in datasets.items()}
	feature_shape = datasets["train"].feature_shape
	in_channels = {"SPECTRAL": 2 * feature_shape, "LSTM": 2 * args.hidden_param, "MODE": 1, "AVERAGE": feature_shape}
	integrators = {"SPECTRAL": None, "LSTM": LSTMInteg(feature_shape, args.hidden_param), "MODE": None, "AVERAGE": None}
	model = ModelWrapper(integrators[args.integrator], classifiers[args.integrator](args.model_config, in_channels[args.integrator], num_classes))
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr = args.learning_rate)
	device = torch.device(args.device)
	model = model.to(device)
	max_acc, best_state = train_val(model, dataloaders, criterion, optimizer, args.epochs, args.patience, device, args.print_interval, consistent_model = classifiers[args.integrator] == NullClassifier)
	print("Best accuracy: ", max_acc)
	save_path = os.path.join(args.save_path, f"video-{get_identifier(args.feature_path)}-{args.integrator}-{max_acc:.4f}.pth")
	torch.save({"model": best_state, "config": args.model_config}, save_path)
	setattr(args, "resume_path", save_path)
	if args.evaluate:
		test(args)

if __name__ == '__main__':
	main()
