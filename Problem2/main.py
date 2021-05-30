import os
import os.path
from os.path import join
import argparse
import torch
import torch.nn as nn
from torch import optim
from dataloader import dataloader
from torch.utils.data import DataLoader
from model import PrototypicalNet
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import time
from tqdm import tqdm
from test_testcase import MiniDataset, GeneratorSampler, worker_init_fn, predict
import sys, csv
import numpy as np
import random

def _parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', type = str)
	parser.add_argument('--load', type = int, default = -1)
	parser.add_argument('--epochs', type = int, default = 30)
	parser.add_argument('--train_n_way', type = int, default = 30)
	parser.add_argument('--valid_n_way', type = int, default = 5)
	parser.add_argument('--n_shot', type = int, default = 1)
	parser.add_argument('--distance_type', type = str, default = 'Euclidean_distance')
	parser.add_argument('--iter_per_epoch', type = int, default = 2000)
	parser.add_argument('--lr_step', type = int, default = 20)
	parser.add_argument('--n_query', type = int, default = 15)
	parser.add_argument('--h_degree', type = int, default = 200)
	parser.add_argument('--train_data_path', type = str, default = '../hw4_data/train/')
	parser.add_argument('--train_data_csv', type = str, default = '../hw4_data/train.csv')
	parser.add_argument('--valid_data_path', type = str, default = '../hw4_data/val/')
	parser.add_argument('--valid_data_csv', type = str, default = '../hw4_data/val.csv')
	parser.add_argument('--testcase_csv', type = str, default = '../hw4_data/val_testcase.csv')
	parser.add_argument('--testcase_pred_csv', type = str, default = './val_testcase_pred.csv')
	parser.add_argument('--testcase_gt_csv', type = str, default = '../hw4_data/val_testcase_gt.csv')
	return parser.parse_args()

def eval(args):
	# read your prediction file
	with open(args.testcase_pred_csv, mode='r') as pred:
		reader = csv.reader(pred)
		next(reader, None)  # skip the headers
		pred_dict = {int(rows[0]): np.array(rows[1:]).astype(int) for rows in reader}
	# read ground truth data
	with open(args.testcase_gt_csv, mode='r') as gt:
		reader = csv.reader(gt)
		next(reader, None)  # skip the headers
		gt_dict = {int(rows[0]): np.array(rows[1:]).astype(int) for rows in reader}
	if len(pred_dict) != len(gt_dict):
		sys.exit("Test case length mismatch.")
	episodic_acc = []
	for key, value in pred_dict.items():
		if key not in gt_dict:
			sys.exit("Episodic id mismatch: \"{}\" does not exist in the provided ground truth file.".format(key))
		episodic_acc.append((gt_dict[key] == value).mean().item())
	episodic_acc = np.array(episodic_acc)
	mean = episodic_acc.mean()
	std = episodic_acc.std()
	print('Accuracy: {:.2f} +- {:.2f} %'.format(mean * 100, 1.96 * std / (600)**(1/2) * 100))
	print()

def write_csv(predictions, pred_path):
	with open(pred_path, "w+") as f:
		header = 'episode_id,query0,query1,query2,query3,query4,query5,query6,query7,query8,query9,query10,query11,query12,query13,query14,query15,query16,query17,query18,query19,query20,query21,query22,query23,query24,query25,query26,query27,query28,query29,query30,query31,query32,query33,query34,query35,query36,query37,query38,query39,query40,query41,query42,query43,query44,query45,query46,query47,query48,query49,query50,query51,query52,query53,query54,query55,query56,query57,query58,query59,query60,query61,query62,query63,query64,query65,query66,query67,query68,query69,query70,query71,query72,query73,query74'
		f.write("%s\n" %(header))
		for index, prediction in enumerate(predictions):
			f.write("%s," %(str(index)))
			prediction = str(prediction)[1:-1].replace(' ', '')
			f.write("%s\n" %(prediction))

def TA_eval(args, device, N_query, model):
	test_dataset = MiniDataset(args.valid_data_csv, args.valid_data_path)
	test_loader = DataLoader(
		test_dataset, batch_size=args.valid_n_way * (N_query + args.n_shot),
		num_workers=3, pin_memory=False, worker_init_fn=worker_init_fn,
		sampler=GeneratorSampler(args.testcase_csv))
	prediction_results = predict(args, device, model, test_loader)
	write_csv(prediction_results, args.testcase_pred_csv)
	if args.mode != 'test': eval(args)

def calculate_distance(args, support_feature, q_feature):
	if args.distance_type == 'Euclidean_distance':
		distance = -(torch.linalg.norm(support_feature - q_feature, dim = 1) ** 2).unsqueeze(0)
	elif args.distance_type == 'Cosine_similarity':
		cos_sim = nn.CosineSimilarity(dim = 1)
		distance = cos_sim(support_feature, q_feature.unsqueeze(0)).unsqueeze(0)
	return distance

def convert2predictions(args, support_feature, query_feature):
    support_feature = support_feature.view(-1, args.n_shot, support_feature.shape[-1])
    support_feature = torch.mean(support_feature, dim = 1).squeeze(1)
    predictions = []
    n_way, channels = support_feature.shape
    query_feature = query_feature.view(-1, n_way, channels)
    for q_feature in query_feature:
        for feature in q_feature:
            distance = calculate_distance(args, support_feature, feature)
            predictions.append(torch.argmax(distance, dim = 1).item())
    return predictions

def convert2distances(args, support_feature, query_feature):
	distances = []
	support_feature = support_feature.view(-1, args.n_shot, support_feature.shape[-1])
	support_feature = torch.mean(support_feature, dim = 1).squeeze(1)
	for q_feature in query_feature:
		distance = calculate_distance(args, support_feature, q_feature)
		distances.append(distance)
	return torch.cat(distances, dim = 0)

def train(args, device):
	torch.multiprocessing.freeze_support()
	if args.mode == 'train':
		train_dataloader = dataloader(args.mode, args.train_n_way, args.n_shot, args.train_data_path, args.train_data_csv, args.iter_per_epoch)
		train_data = DataLoader(train_dataloader, batch_size = 1, num_workers = 5, shuffle = False, pin_memory = True)
		valid_dataloader = dataloader('valid', args.valid_n_way, args.n_shot, args.valid_data_path, args.valid_data_csv, args.iter_per_epoch)
		valid_data = DataLoader(valid_dataloader, batch_size = 1, num_workers = 5, shuffle = False, pin_memory = True)
		print('loading model...')
		PNet = PrototypicalNet(args)
		print(PNet)
		total_params = sum(p.numel() for p in PNet.parameters() if p.requires_grad)
		print("Total number of params = ", total_params)
		PNet.cuda().float()

		save_path = './models_' + args.distance_type + '/'
		os.makedirs(save_path, exist_ok = True)

		if args.load != -1:
			PNet.load_state_dict(torch.load(save_path + str(args.load) + '.ckpt'))

		optimizer = optim.SGD(filter(lambda param : param.requires_grad, list(PNet.parameters())), lr = 1e-3, weight_decay = 0.012, momentum = 0.9)
		scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = args.lr_step * args.iter_per_epoch, gamma = 0.5)
		loss_func = nn.CrossEntropyLoss()
		loss_func.cuda()
		best_loss = 100.0

		train_digits_labels = torch.tensor(sorted([i for i in range(0, args.train_n_way)] * args.n_shot), dtype = torch.long).cuda()
		valid_digits_labels = torch.tensor(sorted([i for i in range(0, args.valid_n_way)] * args.n_shot), dtype = torch.long).cuda()

		for epoch in range(args.load + 1, args.epochs):
			print('epoch: {}  (lr: {})'.format(epoch, scheduler.get_last_lr()[0]))
			PNet.train()
			total_loss = 0
			for index, (support_image, query_image, label) in enumerate(tqdm(train_data, ncols = 70)):
				batch_support_image, batch_query_image, batch_labels = support_image.to(device).squeeze(0), query_image.to(device).squeeze(0), label
				support_feature, origin_support, fake_support = PNet(batch_support_image)
				query_feature, origin_query, fake_query = PNet(batch_query_image)
				distances = convert2distances(args, support_feature, query_feature)
				loss = loss_func(distances, train_digits_labels)
				total_loss += loss.item()
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				scheduler.step()
		
			avg_loss = total_loss / len(train_data)
			print('train_avg_loss: {:.10f}'.format(avg_loss))

			prediction_results = []
			with torch.no_grad():
				PNet.eval()
				total_loss = 0
				for index, (support_image, query_image, label) in enumerate(tqdm(valid_data, ncols = 70)):
					batch_support_image, batch_query_image, batch_labels = support_image.to(device).squeeze(0), query_image.to(device).squeeze(0), label
					support_feature, origin_support, fake_support = PNet(batch_support_image)
					query_feature, origin_query, fake_query = PNet(batch_query_image)
					distances = convert2distances(args, support_feature, query_feature)
					prediction_results.append(convert2predictions(args, support_feature, query_feature))
					loss = loss_func(distances, valid_digits_labels)
					total_loss += loss.item()
			
				avg_loss = total_loss / len(valid_data)
				print('valid_avg_loss: {:.10f}'.format(avg_loss))
				print()
				if avg_loss < best_loss:
					best_loss = avg_loss
					torch.save(PNet.state_dict(), save_path + 'best.ckpt', _use_new_zipfile_serialization = False)
				torch.save(PNet.state_dict(), save_path + '{}.ckpt'.format(epoch), _use_new_zipfile_serialization = False)

				TA_eval(args = args, device = device, N_query = args.n_query, model = PNet)

	elif args.mode == 'valid':
		print('loading model...')
		PNet = PrototypicalNet(args)
		print(PNet)
		total_params = sum(p.numel() for p in PNet.parameters() if p.requires_grad)
		print("Total number of params = ", total_params)
		PNet.cuda().float()

		save_path = './models_' + args.distance_type + '/'
		PNet.load_state_dict(torch.load(save_path + str(args.load) + '.ckpt'))
		PNet.eval()

		TA_eval(args = args, device = device, N_query = args.n_query, model = PNet)

	elif args.mode == 'tsne':
		# train_n_way == 1
		# distance_type == 'Euclidean_distance'
		# h_degree == 1
		train_dataloader = dataloader(args.mode, args.train_n_way, args.n_shot, args.train_data_path, args.train_data_csv, args.iter_per_epoch)
		train_data = DataLoader(train_dataloader, batch_size = 1, num_workers = 5, shuffle = False, pin_memory = True)

		print('loading model...')
		PNet = PrototypicalNet(args)
		print(PNet)
		total_params = sum(p.numel() for p in PNet.parameters() if p.requires_grad)
		print("Total number of params = ", total_params)
		PNet.cuda().float()

		save_path = './models_' + args.distance_type + '/'
		os.makedirs(save_path, exist_ok = True)
		PNet.load_state_dict(torch.load(save_path + str(args.load) + '.ckpt'))

		print('epoch: {}'.format(args.load))
		PNet.eval()

		data_df = pd.read_csv(args.train_data_csv).set_index('id')
		category_originF_fakeF = {label : [[], []] for label in data_df['label']}
		category_counter = {label : 0 for label in data_df['label']}
		selet_category = []

		for index, (support_image, query_image, label) in enumerate(tqdm(train_data, ncols = 70)):
			batch_support_image, batch_query_image, batch_labels = support_image.to(device).squeeze(0), query_image.to(device).squeeze(0), label
			support_feature, origin_support, fake_support = PNet(batch_support_image)
			label = label[0][0]
			if category_counter[label] == 200: continue
			category_originF_fakeF[label][0].append(origin_support.squeeze().cpu().detach().numpy())
			category_originF_fakeF[label][1].append(fake_support.squeeze().cpu().detach().numpy())
			category_counter[label] += 1
			if category_counter[label] == 200:
				selet_category.append(label)
				if len(selet_category) == 5: break

		show_tsne(category_originF_fakeF, selet_category)

def show_tsne(category_originF_fakeF, selet_category):
	labels = np.array(sorted([i for i in range(len(selet_category))] * 200))
	origin_features = np.array([np.array(category_originF_fakeF[category][0], dtype = np.float32) for category in selet_category]).reshape(labels.shape[0], -1)
	fake_features = np.array([np.array(category_originF_fakeF[category][1], dtype = np.float32) for category in selet_category]).reshape(labels.shape[0], -1)

	combine_features = np.concatenate((origin_features, fake_features), axis = 0)
	combine_features_fit = TSNE(n_components = 2, n_jobs = -1).fit_transform(combine_features)
	plt.scatter(combine_features_fit[0:labels.shape[0], 0], combine_features_fit[0:labels.shape[0], 1], c = labels, cmap = plt.cm.jet, s = 10, marker = 'x')
	plt.scatter(combine_features_fit[labels.shape[0]:, 0], combine_features_fit[labels.shape[0]:, 1], c = labels, cmap = plt.cm.jet, s = 10, marker = '^')
	plt.show()

def test(args, device):
	print('loading model...')
	PNet = PrototypicalNet(args)
	print(PNet)
	total_params = sum(p.numel() for p in PNet.parameters() if p.requires_grad)
	print("Total number of params = ", total_params)
	PNet.cuda().float()

	PNet.load_state_dict(torch.load('./Problem2/test.ckpt'))
	PNet.eval()

	TA_eval(args = args, device = device, N_query = args.n_query, model = PNet)

if __name__ == '__main__':
	args = _parse_args()
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	test(args, device) if args.mode == 'test' else train(args, device)
