# Pionniers du TJ, benissiez-moi par votre Esprits Saints!
from torch import nn
from collections import OrderedDict
class MLPClassifier(nn.Module):
	def __init__(self, channels_n_dropout, in_channels, num_classes, **kwargs):
		super().__init__(**kwargs)
		hiddens = OrderedDict()
		for i, (t_channel, t_dropout) in enumerate(channels_n_dropout):
			hiddens[f"linear_{i}"] = nn.Linear(in_channels, t_channel)
			hiddens[f"relu_{i}"] = nn.ReLU()
			hiddens[f"dropout_{i}"] = nn.Dropout(t_dropout)
			in_channels = t_channel
		self.hiddens = nn.Sequential(hiddens)
		self.fc = nn.Linear(in_channels, num_classes)
	def forward(self, x, return_inner = False):
		tx = x = self.hiddens(x)
		x = self.fc(x)
		if return_inner:
			return x, tx
		else:
			return x
