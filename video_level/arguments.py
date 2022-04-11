# Pionniers du TJ, benissiez-moi par votre Esprits Saints!
from base_arguments import get_base_parser
def get_video_parser():
	parser = get_base_parser()
	parser.add_argument("-il", "--integrate-length", type = int, default = 80, help = "The N for spectral representation, the input length for LSTM. In other words, the output length of integrator.")
	parser.add_argument("-hp", "--hidden-param", type = int, default = 256, help = "The resolultion for spectral representation, the hidden size for LSTM.")
	parser.add_argument("-i", "--integrator", type = str, choices = ["LSTM", "SPECTRAL"], default = "SPECTRAL", help = "The integrator")
	return parser