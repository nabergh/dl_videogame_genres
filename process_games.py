import json
import pprint

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


for game in games:
	if 'genres' not in game or game['image'] is None:
		incomplete_games.append(game['id'])
	else:
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

pp.pprint(genres)
pp.pprint(incomplete_games)
pp.pprint(len(complete_games))

with open('complete_games.json', 'w+') as outfile:
	outfile.write(json.dumps(complete_games, indent = 4))