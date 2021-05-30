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

class Encoder(nn.Module):
	def __init__(self, in_channels = 3, hidden = 64, out_channels = 64):
		super(Encoder, self).__init__()
		self.encoder = nn.Sequential(
							Conv_Block(in_channels, hidden, 1),
							Conv_Block(hidden, hidden, 1),
							Conv_Block(hidden, hidden, 1),
							Conv_Block(hidden, out_channels, 1),
					   )
		self.__initialize_weights()

	def forward(self, input):
		output = self.encoder(input).view(input.size(0), -1)
		return output
	
	def __initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				m.weight.data.normal_(0, 1)
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

def weights_init_uniform(m):
	classname = m.__class__.__name__
	# for every Linear layer in a model..
	if classname.find('Linear') != -1:
		# apply a uniform distribution to the weights and a bias=0
		m.weight.data.uniform_(0.0, 1.0)
		m.bias.data.fill_(0)

class HallucinationNet(nn.Module):
	def __init__(self, args, in_channels = 1600, noise_channel = 128, out_channels = 1600):
		super(HallucinationNet, self).__init__()
		self.args = args
		self.encoder = nn.Sequential(
							nn.Linear(in_channels + noise_channel, out_channels // 4),
							nn.Dropout(0.2),
							nn.ReLU(True),
							nn.Linear(out_channels // 4, out_channels),
					   )
		weights_init_uniform(self.encoder)

	def forward(self, feauture_noise):
		fake_feature = self.encoder(feauture_noise)
		return fake_feature

class MLP(nn.Module):
	def __init__(self, in_channels, out_channels, scale = 0.1):
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

class PrototypicalNet(nn.Module):
	def __init__(self, args):
		super(PrototypicalNet, self).__init__()
		self.args = args
		self.extractor = Encoder()
		self.hallucinate = HallucinationNet(args)
		self.mlp = MLP(in_channels = 1600, out_channels = 1600)

	def augment(self, support_feature):
		fake_support_feature = []
		batch_size, channels = support_feature.shape
		for select_index in range(batch_size):
			noise = torch.randn(self.args.h_degree, 128).cuda()
			feauture_noise = torch.cat([support_feature[select_index].unsqueeze(0).expand(self.args.h_degree, -1), noise], dim = 1)
			fake_support_feature.append(self.hallucinate(feauture_noise).unsqueeze(0))
		support_feature = support_feature.unsqueeze(1)
		fake_support_feature = torch.cat(fake_support_feature, dim = 0)
		mix_feature = torch.mean(torch.cat([support_feature, fake_support_feature], dim = 1), dim = 1).squeeze(1)
		return mix_feature, support_feature, fake_support_feature
		
	def forward(self, input):
		feature = self.extractor(input)
		mix_feature, support_feature, fake_support_feature = self.augment(feature)
		mix_feature = self.mlp(mix_feature)
		support_feature = self.mlp(support_feature)
		fake_support_feature = self.mlp(fake_support_feature)
		return mix_feature, support_feature, fake_support_feature