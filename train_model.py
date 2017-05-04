import json
import argparse
import time
from datetime import date
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as data
from torch import optim
from torch.autograd import Variable as V
from genre_classifier import GenreClassifier
from dataset import GameFolder
from tensorboard_logger import configure, log_value

NUM_GENRES = 16


def train(model, args):

    params = np.concatenate([tensor.cpu().numpy().flatten() for _,tensor in model.state_dict().items()])
    print(str(len(params)) + ' parameters to train')

    timestring = str(date.today()) + '_' + time.strftime("%Hh-%Mm-%Ss", time.localtime(time.time()))
    run_name = args.save_name + '_' + timestring
    configure("logs/" + run_name, flush_secs=5)

    games_json = json.load(open(args.games_json, 'r'))

    train_loader = data.DataLoader(
        GameFolder(games_json, 'scraped_imgs/'),
        batch_size=args.batch_size, shuffle=True,
        num_workers=8, pin_memory=True)

    learning_rate = 0.0001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    bce = nn.BCELoss()
    batch_ctr = 0
    
    epoch_loss = 0
    
    print("Gonna train!")
    for epoch in range(args.epochs):
        if epoch > 0:
            last_epoch_loss = epoch_loss
            epoch_loss = 0 

        for i, (imgs, genres, game_id) in enumerate(train_loader):
            prediction = model(V(imgs).type(args.dtype))
            batch_loss = bce(prediction, V(genres).type(args.dtype))

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            log_value('BCE loss', batch_loss.data[0], batch_ctr)
            log_value('Learning rate', optimizer.param_groups[0]['lr'], batch_ctr)

            epoch_loss += batch_loss.data[0]
            batch_ctr += 1


        if epoch % 2 == 0:
            pickle.dump(model.state_dict(), open('models/' + args.save_name + '.p', 'wb'))

        if epoch > 2: #arbitrary epoch choice 
            if (last_epoch_loss - epoch_loss)/epoch_loss < .0001:
                for param in range(len(optimizer.param_groups)):
                    optimizer.param_groups[param]['lr'] = optimizer.param_groups[param]['lr']/2


parser = argparse.ArgumentParser(description='Genre Classifier Train')
parser.add_argument('--batch-size', type=int, default=64,
    help='batch size (default: 64)')
parser.add_argument('--img-dir', type=str,
    help='cover art image directory')
parser.add_argument('--games-json', type=str,
    help='path to json with games data')
parser.add_argument('--save-name', type=str, default='vg_genre_classifier',
    help='file name to save model params dict')
parser.add_argument('--load-name', type=str,
    help='file name to load model params dict')
parser.add_argument('--gpu', action="store_true",
    help='attempt to use gpu')
parser.add_argument('--epochs', type=int, default=9999999999999,
    help='number of epochs to train, defaults to run indefinitely')

if __name__ == '__main__':
    args = parser.parse_args()
    args.dtype = torch.cuda.FloatTensor if torch.cuda.is_available() and args.gpu else torch.FloatTensor

    model = GenreClassifier(args.dtype, NUM_GENRES)
    if args.load_name is not None:
        model.load_state_dict(pickle.load(open(args.load_name + '.p', 'rb')))
    train(model, args)