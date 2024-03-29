# Pionniers du TJ, benissiez-moi par votre Esprits Saints!
from base_arguments import get_base_parser
def get_video_parser():
	parser = get_base_parser()
	parser.add_argument("-il", "--integrate-length", type = int, default = 80, help = "The N for spectral representation, the input length for LSTM. In other words, the output length of integrator.")
	parser.add_argument("-hp", "--hidden-param", type = int, default = 256, help = "The resolultion for spectral representation, the hidden size for LSTM.")
	parser.add_argument("-i", "--integrator", type = str, choices = ["LSTM", "SPECTRAL", "MODE", "AVERAGE"], default = "SPECTRAL", help = "The integrator")
	parser.add_argument("-eva", "--evaluate", action = "store_true", default = False, help = "Whether evaluate the result.")
	parser.add_argument("-catf", "--cache-transform", action = "store_true", default = True, help = "Whether cache the transform")
	parser.add_argument("-si", "--select-interval", type = int, default = 1, help = "Select one frame from how many frames, = 24 / FPS. Alternative for FPS. If FPS = 24, SI = 1, FPS = 12, SI = 2, etc.")
	return parser
