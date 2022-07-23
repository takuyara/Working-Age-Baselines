import os
import sys
sys.path.append("../")
import csv
from utils import get_label
res = {}
def add_csv(path, res):
	with open(path, newline = "") as f:
		reader = csv.reader(f)
		for row in reader:
			site, participant, task, result = tuple(row[ : 4])
			result = int(result)
			res.setdefault((site, participant, task), []).append(result)
add_csv("prediction-valence.csv", res)
add_csv("prediction-arousal.csv", res)

def get_true_label(label_path, site, participant, task, predict_dim):
	csv_path = os.path.join(os.path.join(os.path.join(label_path, site), "SAM"), f"{participant}.csv")
	try:
		with open(csv_path, newline = "") as f:
			reader = csv.DictReader(f)
			for i, row in enumerate(reader):
				if i == predict_dim:
					return get_label(int(row[task]))
	except Exception as e:
		logger.warning(e)

recalls = {}
for (site, participant, task), vap in res.items():
	if len(vap) != 2:
		print("ID Wrong: ", site, participant, task)
	else:
		for predict_dim, y_pred in enumerate(vap):
			y_true = get_true_label("D:\\working-age-data\\questionnaire", site, participant, task, predict_dim)
			recalls.setdefault((predict_dim, y_true), []).append(1 if y_pred == y_true else 0)

recalls_ = {}

for (predict_dim, y_pred), reses in recalls.items():
	recalls_.setdefault(predict_dim, []).append(sum(reses) / len(reses))

for predict_dim, reses in recalls_.items():
	print(predict_dim, sum(reses) / len(reses))
