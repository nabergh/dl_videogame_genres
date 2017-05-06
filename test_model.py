import json
import argparse
import time
from datetime import date
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V
from genre_classifier import GenreClassifier
from dataset import GameFolder
from tensorboard_logger import configure, log_value

NUM_GENRES = 16


def test(model, args):
    
    # timestring = str(date.today()) + '_' + time.strftime("%Hh-%Mm-%Ss", time.localtime(time.time()))
    # run_name = args.load_name.split('/')[-1] + '_test_' + timestring
    # configure("logs/" + run_name, flush_secs=5)

    games_json = json.load(open(args.games_json, 'r'))

    train_loader = data.DataLoader(
        GameFolder(games_json, 'scraped_imgs/'),
        batch_size=args.batch_size, shuffle=True,
        num_workers=8, pin_memory=True)

    bce = nn.BCELoss()
    batch_ctr = 0
    epoch_loss = 0
    
    model.eval()

    preds = None
    labels = None

    print("Gonna test!")
    for i, (imgs, genres, game_id) in enumerate(train_loader):
        prediction = model(V(imgs, volatile=True).type(args.dtype))
        if preds is None:
            preds = prediction.clone().cpu()
            labels = V(genres, volatile=True).cpu()
        else:
            preds = torch.cat((preds, prediction.clone().cpu()), 0)
            labels = torch.cat((labels, V(genres, volatile=True).cpu()), 0)

        batch_loss = bce(prediction, V(genres, volatile=True).type(args.dtype))


        # log_value('BCE loss', batch_loss.data[0], i)

        epoch_loss += batch_loss.data[0]
        batch_ctr += 1
    test_loss = bce(preds, labels)
    print('Test BCE Loss: ' + str(test_loss))

    pickle.dump(preds, open('results/test_preds.p', 'wb'))
    pickle.dump(labels, open('results/test_labels.p', 'wb'))


    # rand_preds = preds.clone()

    # for genre_ind in range(preds.size(1)):
    #     genre_labs = np.random.shuffle(labels.data[:,genre_ind].numpy())
    #     print(genre_labs)
    #     rand_preds.data[:,genre_ind] = torch.FloatTensor(genre_labs)
    # rand_test_loss = bce(rand_preds, labels)
    # print('BCE loss for random predictions: ' + str(rand_test_loss.data[0]))




parser = argparse.ArgumentParser(description='Genre Classifier Train')
parser.add_argument('--batch-size', type=int, default=64,
    help='batch size (default: 32)')
parser.add_argument('--img-dir', type=str,
    help='cover art image directory')
parser.add_argument('--games-json', type=str,
    help='path to json with games data')
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
        model.load_state_dict(pickle.load(open(args.load_name, 'rb')))
        test(model, args)
    else:
        print('You must supply a model to load with the --load-model arg')