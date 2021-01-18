from pbpstats.data_loader import StatsNbaEnhancedPbpLoader, StatsNbaPossessionLoader, StatsNbaShotsLoader, StatsNbaGameFinderLoader

#possession_loader = StatsNbaPossessionLoader("0021900001", "file", "/data")
#print(possession_loader.items[0].data)  # prints dict with the first possession of the game

#pbp_loader = StatsNbaEnhancedPbpLoader("0021900001", "file", "/data")
#print(pbp_loader.items[0].data)  # prints dict with the first event of the game

#shot_loader = StatsNbaShotsLoader("0021900001", "file", "/data")
#print(shot_loader.items[0].data) # prints dict with data for one shot from game

#game_finder_loader = StatsNbaGameFinderLoader("nba", "1983-84", "Regular Season", "f")
#print(game_finder_loader.items[0].data) # prints dict for first game

season_types = ["Regular Season", "Playoffs"]
seasons = [str(i) + "-" + str(i+1)[-2:] for i in range(2000, 2020)]

all_games = []
for season in seasons:
    for season_type in season_types:
        season_games = StatsNbaGameFinderLoader("nba", season, season_type, "file", "/data")
        for game in season_games.items:
            all_games.append(game.data['game_id'])

#print(all_games)
print(len(all_games))

#game_finder_loader.items[0].data
#all_posessions = []
