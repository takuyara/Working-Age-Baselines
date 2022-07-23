# Pionniers du TJ, benissiez-moi par votre Esprits Saints!
import os
import config
import logging
logger = logging.getLogger(__name__)

male_participants = ["SM0RS1110RM29LBA0001A", "BN2NG500LF37LBV0001A", "DR2RS100RM34LBV0001A", "DV3NN700RM32LBV0001A", "GD0NF800RM32LBV0001A", "LT2CM500LM31LBV0001A", "ND1TF1200RM32LBV0001A", "NG2DR100RM28LBV0001A", "PC1TH100RF36LBV0001A", "PR3TC300RM35LBV0001A", "SM0TN800RM29LBV0001A", "VR1NN900LM29LBV0001A", "DJ2MS709RM23LBV0001L", "JK2NN709RM23LBV0001F", "KG0LL309RM22LBV0001F", "LS2NC909RM24LBV0001F", "MB2XD709LM30LBV0001F", "PK2HH409RM23LBV0001F", "TEST_RWTH_MR3RC609RM23LBV0001F", "UCAM-SS001", "UCAM-SS004", "UCAM-SS006", "UCAM-SS007", "UCAM-SS009"]

num_classes = 3
def get_label(score):
	if score in [1, 2, 3]:
		return 0
	if score in [4, 5, 6]:
		return 1
	if score in [7, 8, 9]:
		return 2
	logger.warning(f"Invalid questionnaire score: {score}")

def get_dim_name(x):
	if x == 0:
		return "v"
	if x == 1:
		return "a"
	if x == 2:
		return "d"
	logger.warning(f"Invalid dim input: {x}")

def get_identifier(path):
	__, res = os.path.split(path)
	return res

def get_partitions(feature_path, val_index, gender = "all"):
	partitions = {"train": [], "val": []}
	for i, site in enumerate(config.all_sites):
		site_path = os.path.join(feature_path, site)
		for participant in os.listdir(site_path):
			if gender != "all":
				if (gender == "f") ^ (participant in male_participants):
					continue
			for task in config.all_tasks:
				task_path = os.path.join(os.path.join(site_path, participant), f"{task}.npy")
				if not os.path.exists(task_path):
					# logger.warning(f"Feature file not found: {task_path}")
					continue
				t_part = "val" if i == val_index else "train"
				partitions[t_part].append((site, participant, task))
	return partitions

def get_partitions1(feature_path, __, ___):
	partitions = []
	for site in config.all_sites:
		site_path = os.path.join(feature_path, site)
		if not os.path.exists(site_path):
			continue
		for participant in os.listdir(site_path):
			for task in ["DE01", "DH01", "NBE01", "NBH01"]:
				task_path = os.path.join(os.path.join(site_path, participant), f"{task}.npy")
				if not os.path.exists(task_path):
					logger.warning(f"Feature file not found: {task_path}")
					continue
				partitions.append((site, participant, task))
	return {"val": partitions}

def get_info(path):
	path, task = os.path.split(path)
	path, participant = os.path.split(path)
	path, dataset = os.path.split(path)
	return [dataset, participant, task.replace(".npy", "")]

def get_best_checkpoint_path(base_path, prefix):
	best_accuracy, best_path = 0, None
	for this_path in os.listdir(base_path):
		if this_path.startswith(prefix):
			accuracy = float(this_path[len(prefix) : this_path.find(".pth")])
			if accuracy > best_accuracy:
				best_accuracy, best_path = accuracy, this_path
	if best_path is None:
		print("Warning: checkpoint not found ", prefix)
	return best_path
