# Pionniers du TJ, benissiez-moi par votre Esprits Saints!
import os
import csv
import numpy as np
from torch.utils.data import Dataset
from utils import get_label
import logging
logger = logging.getLogger(__name__)

class FrameDataset(Dataset):
	def __init__(self, feature_path, label_path, subtask_ids, predict_dim):
		super().__init__()
		self.total_items = []
		for site, participant, task in subtask_ids:
			csv_path = os.path.join(os.path.join(os.path.join(label_path, site), "SAM"), f"{participant}.csv")
			try:
				with open(csv_path, newline = "") as f:
					reader = csv.DictReader(f)
					for i, row in enumerate(reader):
						if i == predict_dim:
							this_label = get_label(int(row[task]))
							break
			except Exception as e:
				logger.warning(e)
				continue
			this_feature_path = os.path.join(os.path.join(os.path.join(feature_path, site), participant), f"{task}.npy")
			if not os.path.exists(this_feature_path):
				# logger.warning(f"Feature file not found: {this_feature_path}")
				continue
			n_frames = np.load(this_feature_path).shape[0]
			self.total_items.extend([(this_feature_path, i, this_label) for i in range(n_frames)])
		self.feature_shape = np.load(self.total_items[0][0])[0, ...].flatten().shape[0]
	def __len__(self):
		return len(self.total_items)
	def __getitem__(self, idx):
		feature_path, frame_index, this_label = self.total_items[idx]
		feature_value = np.load(feature_path)[frame_index, ...].flatten()
		return feature_value, this_label

class InferDataset(Dataset):
	def __init__(self, feature_path):
		super().__init__()
		self.total_items = np.load(feature_path)
		self.feature_shape = self.total_items[0, ...].flatten().shape[0]
	def __len__(self):
		return self.total_items.shape[0]
	def __getitem__(self, idx):
		return self.total_items[idx, ...].flatten()
