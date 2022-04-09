# Pionniers du TJ, benissiez-moi par votre Esprits Saints!
import argparse
def get_base_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument("-lp", "--label-path", type = str, default = "D:\\working-age-data\\questionnaire", help = "The path to the root folder of label files.")
	parser.add_argument("-fp", "--feature-path", type = str, default = None, help = "The path to the root folder of features. The default value should be specified in derived parsers.")
	parser.add_argument("-cf", "--cur-fold", type = int, default = 0, choices = list(range(4)), help = "The current fold. Site wise, thus should be in 0 to 3.")
	parser.add_argument("-pd", "--predict-dim", type = int, default = 0, choices = [0, 1], help = "The dimension to predict. 0 for valance and 1 for arousal.")
	parser.add_argument("--save-path", type = str, default = "../checkpoints/", help = "The path to the checkpoint path.")
	parser.add_argument("-cfg", "--model-config", type = str, default = "./model_config.csv", help = "The model config csv file. Headers should be [channels, dropout].")
	
	parser.add_argument("-e", "--epochs", type = int, default = 50, help = "The number of epochs")
	parser.add_argument("-b", "--batch-size", type = int, default = 32, help = "The batch size.")
	parser.add_argument("-p", "--patience", type = int, default = 5, help = "The patience for early stopping.")
	parser.add_argument("-g", "--device", type = str, default = "cuda", help = "The GPU index (or CPU if do not use GPU) to use.")
	parser.add_argument("-lr", "--learning-rate", type = float, default = 1e-3, help = "The learning rate.")

	parser.add_argument("--print-interval", type = int, default = None, help = "The print interval for mini-batches. If not set, use progress bar to show.")
	return parser
