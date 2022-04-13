# Pionniers du TJ, benissiez-moi par votre Esprits Saints!
from torch import nn
import numpy as np
from tqdm import tqdm
from numpy.fft import fft
from math import ceil, floor, pi

class SpectralRepr(nn.Module):
	# N < resolution / 2
	def __init__(self, N, resolution, **kwargs):
		super().__init__(**kwargs)
		self.N = N
		self.resolution = resolution
	
	def cut_data(self, x, resolution):
		n_total = x.shape[0]
		k = floor(n_total / resolution)
		n_res = k * resolution
		n_del = n_total - n_res
		x = x[int(ceil(n_del / 2)) : n_total - int(floor(n_del / 2)), : ]
		return x, k

	def fft_select(self, x, N, k, resolution):
		n_total = x.shape[0]
		complex_fft = fft(x, axis = 0)
		amp_map = np.abs(complex_fft / n_total)
		phase_map = np.angle(complex_fft)
		amp_com = []
		phase_com = []
		for i in range(resolution):
			amp_com.append(amp_map[i * k, : ])
			phase_com.append(phase_map[i * k, : ])
		amp_com = np.array(amp_com)
		phase_com = np.array(phase_com)
		idx = n_total // 2
		amp_map = np.concatenate([amp_map[idx : idx + 1, : ], amp_com[ : N - 1, : ]], axis = 0)
		phase_map = np.concatenate([phase_map[idx : idx + 1, : ], phase_com[ : N - 1, : ]], axis = 0)
		return amp_map, phase_map

	def forward(self, t):
		sp0 = t.shape
		t = t.reshape(t.shape[0], -1)
		t, k = self.cut_data(t, self.resolution)
		t = t - np.median(t, axis = 0)
		amp_map, phase_map = self.fft_select(t, self.N, k, self.resolution)
		t = np.concatenate([amp_map, phase_map], axis = 1)
		return t

class LSTMInteg(nn.Module):
	def __init__(self, in_channels, hidden_size, **kwargs):
		super().__init__(**kwargs)
		self.lstm = nn.LSTM(input_size = in_channels, hidden_size = hidden_size, num_layers = 1, bidirectional = True, batch_first = True)
	def forward(self, x):
		x, __ = self.lstm(x)
		return x
