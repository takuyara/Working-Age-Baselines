# Pionniers du TJ, benissiez-moi par votre Esprits Saints!
from torch import nn
from collections import OrderedDict

class CNNClassifier(nn.Module):
	def __init__(self, kernels_channels_dropout, in_channels, num_classes, **kwargs):
		super().__init__(**kwargs)
		hiddens = OrderedDict()
		for i, (kernels, channels, dropout) in enumerate(kernels_channels_dropout):
			hiddens[f"conv_{i}"] = nn.Conv1d(in_channels, channels, kernels)
			hiddens[f"bnorm_{i}"] = nn.BatchNorm1d(channels)
			hiddens[f"relu_{i}"] = nn.ReLU()
			hiddens[f"dropout_{i}"] = nn.Dropout(dropout)
			in_channels = channels
		self.hiddens = nn.Sequential(hiddens)
		self.before_fc = nn.Sequential(nn.AdaptiveMaxPool1d(1), nn.Flatten(), nn.Dropout(0.3), nn.ReLU())
		self.fc = nn.Linear(in_channels, num_classes)
	def forward(self, x):
		x = self.hiddens(x)
		x = self.before_fc(x)
		x = self.fc(x)
		return x
