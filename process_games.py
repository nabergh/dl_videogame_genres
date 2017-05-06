import json
import pprint
from PIL import Image
import numpy as np
import os

pp = pprint.PrettyPrinter(indent=4)
games = json.load(open('giantbomb_games_genres.json', 'r'))
genre_map = {'Baseball': 'Sports',
    'Basketball': 'Sports',
    'Billiards': 'Sports',
    'Block-Breaking': 'Arcade',
    'Bowling': 'Sports',
    'Boxing': 'Sports',
    'Brawler': 'Fighting',
    'Card Game': 'Trivia/Board Game',
    'Cricket': 'Sports',
    'Dual-Joystick Shooter': 'Shooter',
    'First-Person Shooter': 'Shooter',
    'Fishing': 'Sports',
    'Fitness': 'Educational',
    'Flight Simulator': 'Simulation',
    'Football': 'Sports',
    'Gambling': 'Trivia/Board Game',
    'Golf': 'Sports',
    'Hockey': 'Sports',
    'Light-Gun Shooter': 'Shooter',
    'MMORPG': 'Role-Playing',
    'MOBA': 'Strategy',
    'Minigame Collection': 'Compilation',
    'Pinball': 'Arcade',
    'Real-Time Strategy': 'Strategy',
    "Shoot 'Em Up": 'Shooter',
    'Skateboarding': 'Sports',
    'Snowboarding/Skiing': 'Sports',
    'Soccer': 'Sports',
    'Surfing': 'Sports',
    'Tennis': 'Sports',
    'Text Adventure': 'Adventure',
    'Track & Field': 'Sports',
    'Vehicular Combat': 'Action',
    'Wrestling': 'Sports'}
genres = {}
incomplete_games = []
complete_games = []

scraped_img_files = os.listdir('scraped_imgs/')
scraped_imgs = {}
for file in scraped_img_files:
    scraped_imgs[file.split('.')[0]] = int(file.split('.')[0])

for game in games:
    if 'genres' not in game or game['image'] is None:
        incomplete_games.append(game['id'])
    else:
        if str(game['id']) in scraped_imgs:
            comp_game = {}
            comp_game['id'] = game['id']
            comp_game['name'] = game['name']
            comp_game['genres'] = []
            for genre in game['genres']:
                if genre['name'] in genre_map:
                    comp_game['genres'].append(genre_map[genre['name']])
                elif genre['name'] == 'Action-Adventure':
                    comp_game['genres'].append('Action')
                    comp_game['genres'].append('Adventure')
                else:
                    comp_game['genres'].append(genre['name'])
            comp_game['genres'] = list(set(comp_game['genres']))
            for genre in comp_game['genres']:
                if genre in genres:
                    genres[genre] += 1
                else:
                    genres[genre] = 1
            complete_games.append(comp_game)

# train_set = []
# test_set = []
# test_ind = np.random.choice(range(len(complete_games)), len(complete_games) / 10, replace = False)

# for i, game in enumerate(complete_games):
#     if i in test_ind:
#         test_set.append(game)
#     else:
#         train_set.append(game)

# pp.pprint(genres)
# pp.pprint(incomplete_games)
pp.pprint(len(complete_games))

with open('complete_games2.json', 'w') as outfile:
    outfile.write(json.dumps(complete_games, indent = 4))

# with open('train_set_games.json', 'w+') as outfile:
#     outfile.write(json.dumps(train_set, indent = 4))

# with open('test_set_games.json', 'w+') as outfile:
#     outfile.write(json.dumps(test_set, indent = 4))
