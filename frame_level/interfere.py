# Pionniers du TJ, benissiez-moi par votre Esprits Saints!
import sys
sys.path.append("../")

from dataset import InferDataset
from base_arguments import get_base_parser
from utils import num_classes
import os
import numpy as np
import torch
import config
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from model import MLPClassifier

logger = logging.getLogger(__name__)

def get_args():
	parser = get_base_parser()
	parser.add_argument("-rp", "--resume-path", type = str, default = None)
	parser.add_argument("-nfp", "--new-feature-path", type = str, default = None)
	parser.add_argument("-npp", "--new-prediction-path", type = str, default = None)
	args = parser.parse_args()
	return args

def solve_single(path, nfp, npp, args):
	if os.path.exists(npp) and os.path.exists(nfp):
		return
	dataset = InferDataset(path)
	dataloader = DataLoader(dataset, batch_size = args.batch_size)
	input_channels = dataset.feature_shape
	resume_data = torch.load(args.resume_path)
	device = torch.device(args.device)
	model = MLPClassifier(resume_data["config"], input_channels, num_classes)
	model.load_state_dict(resume_data["model"])
	model = model.to(device)
	model.eval()
	resy = []
	resf = []
	for x in dataloader:
		x = x.to(device)
		with torch.no_grad():
			y, f = model(x, return_inner = True)
			__, y = torch.max(y, dim = 1)
			y = y.cpu().numpy()
			f = f.cpu().numpy()
			resy.append(y)
			resf.append(f)
	resy = np.concatenate(resy, axis = 0).astype("float32")
	resf = np.concatenate(resf, axis = 0).astype("float32")
	np.save(npp, resy)
	np.save(nfp, resf)

def main():
	args = get_args()
	all_files = []
	for site in config.all_sites:
		sp = os.path.join(args.feature_path, site)
		for participant in os.listdir(sp):
			nfp = os.path.join(os.path.join(args.new_feature_path, site), participant)
			npp = os.path.join(os.path.join(args.new_prediction_path, site), participant)
			os.makedirs(nfp, exist_ok = True)
			os.makedirs(npp, exist_ok = True)
			for task in config.all_tasks:
				tp = os.path.join(os.path.join(sp, participant), f"{task}.npy")
				if not os.path.exists(tp):
					logger.warning(f"Feature file not found: {tp}")
					continue
				all_files.append((tp, os.path.join(nfp, f"{task}.npy"), os.path.join(npp, f"{task}.npy")))
	for tfile, nfp, npp in tqdm(all_files):
		solve_single(tfile, nfp, npp, args)

if __name__ == '__main__':
	main()
