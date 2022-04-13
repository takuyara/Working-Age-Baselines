# Pionniers du TJ, benissiez-moi par votre Esprits Saints!
import os
import csv
from torch.utils.data import Dataset
import numpy as np
import logging
from utils import get_label
logger = logging.getLogger(__name__)

class VideoDataset(Dataset):
	def __init__(self, feature_path, label_path, subtask_ids, predict_dim, min_len, transform = None, required_task = None, cache_transform = False):
		super().__init__()
		self.total_items = []
		self.transform = transform
		self.cache_transform = cache_transform
		for site, participant, task in subtask_ids:
			if required_task is not None:
				if task != required_task:
					continue
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
			this_feature = np.load(this_feature_path)
			if this_feature.shape[0] < min_len:
				# logger.warning(f"Too short input: {this_feature_path}")
				continue
			self.feature_shape = this_feature[0, ...].flatten().shape[0]
			if cache_transform:
				if transform is not None:
					this_feature = transform(this_feature)
			else:
				this_feature = this_feature_path
			self.total_items.append((this_feature, this_label))
	def __len__(self):
		return len(self.total_items)
	def __getitem__(self, idx):
		feature_path, this_label = self.total_items[idx]
		if self.cache_transform:
			feature_value = feature_path
		else:
			feature_value = np.load(feature_path)
			if self.transform is not None:
				feature_value = self.transform(feature_value)
		feature_value = feature_value.reshape(feature_value.shape[0], -1)
		return feature_value.astype("float32"), this_label
