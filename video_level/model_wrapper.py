# Pionniers du TJ, benissiez-moi par votre Esprits Saints!
from torch import nn
class ModelWrapper(nn.Module):
	def __init__(self, integrator, classifier, **kwargs):
		super().__init__(**kwargs)
		self.integrator = integrator
		self.classifier = classifier
	def forward(self, x):
		if self.integrator is not None:
			x = self.integrator(x)
		x = self.classifier(x)
		return x
