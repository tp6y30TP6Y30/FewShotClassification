import os
import sys
import argparse

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler

import csv
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

from PIL import Image
# filenameToPILImage = lambda x: Image.open(x)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
np.random.seed(SEED)

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# mini-Imagenet dataset
class MiniDataset(Dataset):
    def __init__(self, csv_path, data_dir):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path).set_index("id")

        self.transform = transforms.Compose([
            transforms.Lambda(self.openImage),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def openImage(self, x):
        return Image.open(x)

    def __getitem__(self, index):
        path = self.data_df.loc[index, "filename"]
        label = self.data_df.loc[index, "label"]
        image = self.transform(os.path.join(self.data_dir, path))
        return image, label

    def __len__(self):
        return len(self.data_df)

class GeneratorSampler(Sampler):
    def __init__(self, episode_file_path):
        episode_df = pd.read_csv(episode_file_path).set_index("episode_id")
        self.sampled_sequence = episode_df.values.flatten().tolist()

    def __iter__(self):
        return iter(self.sampled_sequence) 

    def __len__(self):
        return len(self.sampled_sequence)

def calculate_distance(args, support_feature, query_feature):
    if args.distance_type == 'Euclidean_distance':
        distance = -(torch.linalg.norm(support_feature - query_feature, dim = 1) ** 2).unsqueeze(0)
    elif args.distance_type == 'Cosine_similarity':
        cos_sim = nn.CosineSimilarity(dim = 1)
        distance = cos_sim(support_feature, query_feature.unsqueeze(0)).unsqueeze(0)
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

def predict(args, device, model, data_loader):
    prediction_results = []
    with torch.no_grad():
        # each batch represent one episode (support data + query data)
        for i, (data, target) in enumerate(tqdm(data_loader, ncols = 70)):
            # split data into support and query data
            support_input = data[:args.valid_n_way * args.n_shot,:,:,:].to(device)
            query_input   = data[args.valid_n_way * args.n_shot:,:,:,:].to(device)
            # create the relative label (0 ~ valid_n_way-1) for query data
            label_encoder = {target[i * args.n_shot] : i for i in range(args.valid_n_way)}

            # query_label = torch.cuda.LongTensor([label_encoder[class_name] for class_name in target[args.valid_n_way * args.n_shot:]])

            # TODO: extract the feature of support and query data
            support_feature = model(support_input)
            query_feature = model(query_input)

            # TODO: calculate the prototype for each class according to its support data
            # TODO: classify the query data depending on the its distense with each prototype
            prediction_results.append(convert2predictions(args, support_feature, query_feature))

    return prediction_results

def parse_args():
    parser = argparse.ArgumentParser(description="Few shot learning")
    parser.add_argument('--N-way', default=5, type=int, help='N_way (default: 5)')
    parser.add_argument('--N-shot', default=1, type=int, help='N_shot (default: 1)')
    parser.add_argument('--N-query', default=15, type=int, help='N_query (default: 15)')
    parser.add_argument('--load', type=str, help="Model checkpoint path")
    parser.add_argument('--test_csv', type=str, help="Testing images csv file")
    parser.add_argument('--test_data_dir', type=str, help="Testing images directory")
    parser.add_argument('--testcase_csv', type=str, help="Test case csv")
    parser.add_argument('--output_csv', type=str, help="Output filename")

    return parser.parse_args()

if __name__=='__main__':
    args = parse_args()

    test_dataset = MiniDataset(args.test_csv, args.test_data_dir)

    test_loader = DataLoader(
        test_dataset, batch_size=args.N_way * (args.N_query + args.N_shot),
        num_workers=3, pin_memory=False, worker_init_fn=worker_init_fn,
        sampler=GeneratorSampler(args.testcase_csv))

    # TODO: load your model

    prediction_results = predict(args, model, test_loader)

    # TODO: output your prediction to csv
