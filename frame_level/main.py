# Pionniers du TJ, benissiez-moi par votre Esprits Saints!
import sys
sys.path.append("../")

import os
import csv
import time
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
from copy import deepcopy
import logging
logger = logging.getLogger(__name__)

from model import MLPClassifier
from dataset import FrameDataset
from utils import get_partitions, num_classes, get_identifier, get_dim_name
from base_arguments import get_base_parser

def get_args():
	parser = get_base_parser()
	args = parser.parse_args()
	path = args.model_config
	r_config = []
	with open(path, newline = "") as f:
		reader = csv.DictReader(f)
		for row in reader:
			r_config.append((int(row["channels"]), float(row["dropout"])))
	args.model_config = r_config
	return args

def get_mean(x):
	return sum(x) / len(x)

def train_val(model, dataloaders, criterion, optimizer, epochs, patience, device, print_interval, ckpt_path = None):
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
			for i, (x, y) in enumerate(pbar):
				optimizer.zero_grad()
				x, y = x.to(device), y.to(device)
				with torch.set_grad_enabled(phase == "train"):
					yp = model(x)
					b_loss = criterion(yp, y)
					if phase == "train":
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
				if (print_interval is not None and (i + 1) % print_interval == 0) or i == len(pbar) - 1:
					print(f"[{i + 1}/{len(pbar)}]: loss = {get_mean(e_loss):.4f}, acc = {get_mean(e_acc) * 100:.2f}", flush = True)
			e_loss = sum(e_loss) / len(e_loss)
			e_acc = sum(e_acc) / len(e_acc)
			if phase == "val":
				if e_loss < min_loss:
					min_loss, min_epoch = e_loss, i
				elif i - min_epoch >= patience:
					stop_training = True
				if e_acc > max_acc:
					max_acc = e_acc
					print("new_max_acc ", e_acc, flush = True)
					best_state = deepcopy(model.state_dict())
					if ckpt_path is not None:
						torch.save(best_state, ckpt_path)
		print(f"Epoch time: {time.time() - sttime}s", flush = True)
		if stop_training:
			break
	return max_acc, best_state

def main():
	args = get_args()
	partitions = get_partitions(args.feature_path, args.cur_fold)
	datasets = {phase : FrameDataset(args.feature_path, args.label_path, partitions[phase], args.predict_dim) for phase in ["train", "val"]}
	dataloaders = {phase : DataLoader(t_dataset, batch_size = args.batch_size, shuffle = True) for phase, t_dataset in datasets.items()}
	in_channels = datasets["train"].feature_shape
	model = MLPClassifier(args.model_config, in_channels, num_classes)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr = args.learning_rate)
	device = torch.device(args.device)
	model = model.to(device)
	ckpt_path = os.path.join(args.save_path, f"frame-{get_identifier(args.feature_path)}-{args.cur_fold}-{get_dim_name(args.predict_dim)}-ckpt.pth")
	max_acc, best_state = train_val(model, dataloaders, criterion, optimizer, args.epochs, args.patience, device, args.print_interval, ckpt_path)
	print("Best accuracy: ", max_acc)
	save_path = os.path.join(args.save_path, f"frame-{get_identifier(args.feature_path)}-{args.cur_fold}-{get_dim_name(args.predict_dim)}-{max_acc:.4f}.pth")
	torch.save({"model": best_state, "config": args.model_config}, save_path)
	os.remove(ckpt_path)

if __name__ == '__main__':
	main()

