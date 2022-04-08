# Pionniers du TJ, benissiez-moi par votre Esprits Saints!
import os
import csv
from torch.utils.data import Dataset
import numpy as np
import logging
from utils import get_label
logger = logging.getLogger(__name__)

class VideoDataset(Dataset):
	def __init__(self, feature_path, label_path, subtask_ids, predict_dim, min_len, transform = None):
		super().__init__()
		self.total_items = []
		self.transform = transform
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
				logger.warning(f"Feature file not found: {this_feature_path}")
				continue
			if np.load(this_feature_path).shape[0] < min_len:
				logger.warning(f"Too short input: {this_feature_path}")
				continue
			self.total_items.append((this_feature_path, this_label))
		self.feature_shape = np.load(self.total_items[0][0])[0, ...].flatten().shape[0]
	def __len__(self):
		return len(self.total_items)
	def __getitem__(self, idx):
		feature_path, this_label = self.total_items[idx]
		feature_value = np.load(feature_path)
		if self.transform is not None:
			feature_value = self.transform(feature_value)
		return feature_value.astype("float32"), this_label
