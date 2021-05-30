import torch.nn as nn
import torch

class Conv_Block(nn.Module):
	def __init__(self, in_channels, out_channels, padding):
		super(Conv_Block, self).__init__()
		self.conv = nn.Sequential(
						nn.Conv2d(in_channels, out_channels, 3, padding = padding),
						nn.BatchNorm2d(out_channels),
						nn.ReLU(True),
						nn.MaxPool2d(2),
					)
	def forward(self, input):
		return self.conv(input)

class Extractor(nn.Module):
	def __init__(self, in_channels = 3, hidden = 64, out_channels = 64):
		super(Extractor, self).__init__()
		self.encoder = nn.Sequential(
							Conv_Block(in_channels, hidden, 1),
							Conv_Block(hidden, hidden, 1),
							Conv_Block(hidden, hidden, 1),
							Conv_Block(hidden, out_channels, 1),
					   )
		self.mlp = MLP()
		self.__initialize_weights()
	
	def __initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				m.weight.data.normal_(0, 1)
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	def forward(self, input):
		feature = self.encoder(input).view(input.size(0), -1)
		feature = self.mlp(feature)
		return feature

class MLP(nn.Module):
	def __init__(self, in_channels = 1600, out_channels = 1600, scale = 0.1):
		super(MLP, self).__init__()
		self.scale = scale
		self.enhance = nn.Sequential(
							nn.Linear(in_channels, in_channels // 4),
							nn.ReLU(True),
							nn.Linear(in_channels // 4, in_channels),
							nn.Dropout(0.5),
						 )
		self.transform = nn.Sequential(
							nn.ReLU(True),
							nn.Linear(in_channels, out_channels),
			  			 )
		weights_init_uniform(self.enhance)
		weights_init_uniform(self.transform)

	def forward(self, input):
		enhance = self.enhance(input)
		output = self.transform(input + self.scale * enhance)
		return output

def weights_init_uniform(m):
	classname = m.__class__.__name__
	# for every Linear layer in a model..
	if classname.find('Linear') != -1:
		# apply a uniform distribution to the weights and a bias=0
		m.weight.data.uniform_(0.0, 1.0)
		m.bias.data.fill_(0)

class Generator(nn.Module):
	def __init__(self, in_channels = 1600, noise_channel = 128, out_channels = 1600):
		super(Generator, self).__init__()
		self.generate = nn.Sequential(
							nn.Linear(in_channels + noise_channel, in_channels // 4),
							nn.ReLU(True),
							nn.Linear(in_channels // 4, out_channels),
					   )
		weights_init_uniform(self.generate)

	def forward(self, feautures):
		generate_size = 2
		noises = torch.randn(feautures.shape[0], generate_size, 128).cuda()
		fake_features = [self.generate(torch.cat([feautures[index].unsqueeze(0).expand(generate_size, -1), noises[index]], dim = 1)) for index in range(len(feautures))]
		fake_features = torch.cat(fake_features, dim = 0)
		return fake_features, noises

class Discriminator(nn.Module):
	def __init__(self, in_channels = 1600, out_channels = 1):
		super(Discriminator, self).__init__()
		self.judge = nn.Sequential(
						nn.Linear(in_channels, in_channels // 4),
						nn.LeakyReLU(0.2),
						nn.Linear(in_channels // 4, out_channels),
						nn.Sigmoid()
					 )
		weights_init_uniform(self.judge)

	def forward(self, input):
		output = self.judge(input)
		return output