from pbpstats.data_loader import StatsNbaEnhancedPbpLoader, StatsNbaPossessionLoader, StatsNbaShotsLoader, StatsNbaGameFinderLoader
from collections import defaultdict
import json
import yaml

with open("team_ids.json", "r") as f:
    team_ids = json.load(f)

with open("player_info.json", "r") as f:
    player_info = json.load(f)

def name(player_id):
    if str(player_id) in player_info:
        return player_info[str(player_id)]['PLAYER_SLUG']

def teamname(team_id):
    if str(team_id) in team_ids:
        return team_ids[str(team_id)]

def accumulate_dicts(d1, d2):
    for k, v in d2.items():
        if k not in d1:
            d1[k] = v
        if type(v) == dict:
            d1[k] = accumulate_dicts(d1[k], v)
        else:
            d1[k] += v
    return d1[k]

class Game():
    def __init__(self, game):
        self.game_id = game['game_id']
        self.date = game['date']
        self.home_team_id = game['home_team_id']
        self.home_team_name = team_ids[str(game['home_team_id'])]
        self.visitor_team_id = game['visitor_team_id']
        self.visitor_team_name = team_ids[str(game['visitor_team_id'])]
        self.plays = defaultdict(lambda: None)
        self.shots = defaultdict(lambda: None)
        self.load_shots()
        self.load_plays()
    
    def load_shots(self):
        shotloader = StatsNbaShotsLoader(self.game_id, "file", "/data")
        for shot in shotloader.items:
            self.shots[shot.data['game_event_id']] = Shot(shot.data)

    def load_plays(self):
        playloader = StatsNbaPossessionLoader(self.game_id, "file", "/data")
        for play in playloader.items:
            p = Play(play, shots=self.shots)
            self.plays[p.play_id] = p

    def __repr__(self):
        small_dict = self.__dict__
        small_dict.pop('plays')
        small_dict.pop('shots')
        return yaml.dump({self.game_id: small_dict}, allow_unicode=True, default_flow_style=False)

class Play():
    def __init__(self, play, shots=defaultdict(lambda: None), players_on_court={}):
        self.play_id = str(play.data['period'] * 1000 + play.data['number'])
        self.start_time = play.start_time
        self.end_time = play.end_time
        self.offense_team = teamname(play.offense_team_id)
        self.possession_start_type = play.possession_start_type
        stats = play.possession_stats
        tot_stats = {}
        for s in stats:
            s[s['stat_key']] = s['stat_value']
            accumulate_dicts(tot_stats, s)
        print(yaml.dump(tot_stats))
        self.events = []
        self.load_events(play.events, shots)
        self.parse_events()

    def load_events(self, events, shots):
        for event in events:
            event_id = event.data['event_num']
            shot = shots[event_id]
            exit()
            #self.events.append(Event(event.data, shot=shot))

    def parse_events(self):
        for event in self.events:
            pass

    def __repr__(self):
        return yaml.dump({self.game_id: self.__dict__}, allow_unicode=True, default_flow_style=False)

class Event():
    def __init__(self, event, shot=None):
        self.event_id = event['event_num']
        self.clock = event['clock']
        self.description = event['description']
        self.event_type_enum = event['event_type']
        self.event_action_type_enum = event['event_action_type']
        self.team = teamname(event['team_id'])
        self.shot = shot
        self.event_type = None
        self.box_change = {}

        if self.event_type_enum == 1:
            self.event_type = 'field_goal_made'
            self.parse_field_goal(event)

        if self.event_type_enum == 2:  
            self.event_type = 'field_goal_missed'
            self.parse_field_goal(event)

        if self.event_type_enum == 3:  
            self.event_type = 'free_throw'
            self.shooter = name(event['player1_id'])

        if self.event_type_enum == 4:  
            self.event_type = 'defensive_rebound'
            self.rebounder = name(event['player1_id'])

        if self.event_type_enum == 5:  
            self.event_type = 'turnover'
            self.turnoverer = name(event['player1_id'])
            self.stealer = name(event['player3_id']) if 'player3_id' in event else None

        if self.event_type_enum == 6:  
            self.event_type = 'foul'
            self.fouler = name(event['player1_id'])
            self.foulee = name(event['player3_id']) if 'player3_id' in event else None

        if self.event_type_enum == 7:  
            self.event_type = 'violation'
            self.violator = name(event['player1_id'])

        if self.event_type_enum == 8:  
            self.event_type = 'substitution'
            self.sub_in = name(event['player1_id'])
            self.sub_out = name(event['player2_id']) 

        if self.event_type_enum == 9: 
            self.event_type = 'timeout'

        if self.event_type_enum == 10: 
            self.event_type = 'jump_ball'
            self.jump_winner = name(event['player1_id'])
            self.tipped_to = name(event['player1_id'])
            self.jump_loser = name(event['player3_id'])
        
        if self.event_type_enum == 11: 
            self.event_type = 'ejection'
            self.ejected = name(event['player1_id'])
        
        if self.event_type_enum == 12: 
            self.event_type = 'period_start'
            self.starters = event['period_starters']
            
        if self.event_type_enum == 13: 
            self.event_type = 'period_end'

    def parse_field_goal(self, event):
        self.shooter = name(event['player1_id'])
        self.assister = name(event['player2_id']) if 'player2_id' in event else None
        self.blocker = name(event['player3_id']) if 'player3_id' in event else None

    def __repr__(self):
        return yaml.dump({self.event_id: self.__dict__}, allow_unicode=True, default_flow_style=False)

class Shot():
    def __init__(self, shot):
        self.event_id = shot['game_event_id'] 
        self.player_id = name(shot['player_id'])
        self.team_id = teamname(shot['team_id'])
        self.period = shot['period']
        self.minutes_remaining = shot['minutes_remaining'] + shot['seconds_remaining'] / 60
        self.type = shot['shot_type']
        self.type_detailed = shot['action_type']
        self.event_type = shot['event_type']
        self.zone = shot['shot_zone_basic']
        self.area = shot['shot_zone_area']
        self.distance = shot['shot_distance']
        self.x = shot['loc_x']
        self.y = shot['loc_y']
        self.attempted = shot['shot_attempted_flag']
        self.made = shot['shot_made_flag']

    def __repr__(self):
        return yaml.dump({self.event_id: self.__dict__}, allow_unicode=True, default_flow_style=False)

season_types = ["Regular Season", "Playoffs"]
seasons = [str(i) + "-" + str(i+1)[-2:] for i in range(2019, 2020)]

all_games = []
for season in seasons:
    for season_type in season_types:
        season_games = StatsNbaGameFinderLoader("nba", season, season_type, "file", "/data")
        for game in season_games.items:
            all_games.append(game.data)

for game in all_games:
    g = Game(game)
    break
#print(g)
        