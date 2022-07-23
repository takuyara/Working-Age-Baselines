import csv
rows = []
with open("prediction-arousal-01.csv", "r") as f:
	for row in f.readlines():
		row = row.split()
		rows.append(row)
with open("prediction-arousal-01-1.csv", "w", newline = "") as f:
	writer = csv.writer(f)
	writer.writerows(rows)