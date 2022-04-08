# Pionniers du TJ, benissiez-moi par votre Esprits Saints!
import os
import config
import logging
logger = logging.getLogger(__name__)

num_classes = 3
def get_label(score):
	if score in [1, 2, 3]:
		return 0
	if score in [4, 5, 6]:
		return 1
	if score in [7, 8, 9]:
		return 2
	logger.warning(f"Invalid questionnaire score: {score}")

def get_identifier(path):
	__, res = os.path.split(path)
	return res

def get_partitions(feature_path, val_index):
	partitions = {"train": [], "val": []}
	for i, site in enumerate(config.all_sites):
		site_path = os.path.join(feature_path, site)
		for participant in os.listdir(site_path):
			for task in config.all_tasks:
				task_path = os.path.join(os.path.join(site_path, participant), f"{task}.npy")
				if not os.path.exists(task_path):
					logger.warning(f"Feature file not found: {task_path}")
					continue
				t_part = "val" if i == val_index else "train"
				partitions[t_part].append((site, participant, task))
	return partitions
