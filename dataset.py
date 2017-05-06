import os
import json
import re
import pickle
import nltk
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from random import randint
from PIL import Image

NUM_GENRES = 16

genre_ids = {'Action': 0,
    'Adventure': 1,
    'Arcade': 2,
    'Compilation': 3,
    'Driving/Racing': 4,
    'Educational': 5,
    'Fighting': 6,
    'Music/Rhythm': 7,
    'Platformer': 8,
    'Puzzle': 9,
    'Role-Playing': 10,
    'Shooter': 11,
    'Simulation': 12,
    'Sports': 13,
    'Strategy': 14,
    'Trivia/Board Game': 15}


def default_loader(path):
    return Image.open(path).convert('RGB')

def make_dataset(games_json : list, img_dir, file_ext):
    games_list = []
    for game in games_json:
        path = os.path.join(img_dir, str(game['id']) + file_ext)
        item = (path, game['genres'], game['id'])
        games_list.append(item)
    print('Number of training games: ' + str(len(games_list)))
    return games_list

def genre_transform(genres):
    gen_tensor = torch.zeros(16)
    for genre in genres:
        gen_tensor[genre_ids[genre]] = 1
    return gen_tensor
       
class GameFolder(data.Dataset):
    def __init__(self, games_json, img_dir, file_ext='.jpeg', transform=None, title_transform=None,
                 loader=default_loader):
        games = make_dataset(games_json, img_dir, file_ext)
        
        self.img_dir = img_dir
        self.games = games
        self.im_trans = transforms.Compose([transforms.Scale(256),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
            # transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])])
            ])
        self.genre_transform = genre_transform
        self.loader = loader

    def __getitem__(self, index):
        path, genres, game_id = self.games[index]
        img = self.loader(path)
        smaller = min(img.size[0], img.size[1])
        larger = max(img.size[0], img.size[1])
        pad_width = larger - smaller
        new_im = None
        if img.size[0] == smaller:
            new_size = (pad_width + img.size[0], img.size[1])
            new_im = Image.new("RGB", new_size)
            new_im.paste(img, (randint(0, pad_width), 0))
        else:
            new_size = (img.size[0], pad_width + img.size[1])
            new_im = Image.new("RGB", new_size)
            new_im.paste(img, (0, randint(0, pad_width)))
        img = new_im
        img = self.im_trans(img)
        genres = genre_transform(genres)
        return img, genres, torch.LongTensor([game_id])

    def __len__(self):
        return len(self.games)