import os
from os import listdir
from os.path import join
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import pandas as pd
import random

class dataloader(Dataset):
	def __init__(self, mode, n_way, n_shot, data_path, data_csv, iter_per_epoch):
		super(dataloader, self).__init__()
		self.mode = mode
		self.n_way = n_way
		self.n_shot = n_shot
		self.iter_per_epoch = iter_per_epoch
		self.data_path = data_path
		self.data_df = pd.read_csv(data_csv).set_index('id')
		self.category_filenames = {label : [] for label in self.data_df['label']}
		for index in range(len(self.data_df)):
			self.category_filenames[self.data_df.loc[index, 'label']].append(self.data_df.loc[index, 'filename'])
		self.transform = transforms.Compose([
							transforms.Lambda(self.openImage),
							transforms.RandomHorizontalFlip(p = 0.2),
							transforms.ToTensor(),
							transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
						 ])

	def __len__(self):
		return self.iter_per_epoch if self.mode == 'train' else len(self.data_df)

	def openImage(self, x):
		return Image.open(x)

	def __getitem__(self, index):
		labels = random.choices(list(self.category_filenames.keys()), k = self.n_way)
		filenames = [random.choices(self.category_filenames[category], k = self.n_shot * 2) for category in labels]
		support_filenames = [filename[:self.n_shot] for filename in filenames]
		query_filenames = [filename[self.n_shot:] for filename in filenames]
		support_images = torch.cat([self.transform(os.path.join(self.data_path, i)).unsqueeze(0) for filename in support_filenames for i in filename], dim = 0)
		query_images = torch.cat([self.transform(os.path.join(self.data_path, i)).unsqueeze(0) for filename in query_filenames for i in filename], dim = 0)
		return support_images, query_images, labels

if __name__ == '__main__':
	test = dataloader('train', 5, 1, '../hw4_data/train/', '../hw4_data/train.csv')
	test_data = DataLoader(test, batch_size = 1, shuffle = False)
	for index, (support_images, query_images, label) in enumerate(test_data):
		print(index, support_images.shape, query_images.shape, label)
		break
