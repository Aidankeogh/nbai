from pbpstats.data_loader import StatsNbaEnhancedPbpLoader, StatsNbaPossessionLoader, StatsNbaShotsLoader, StatsNbaGameFinderLoader
from collections import defaultdict
import json
import yaml
import os
from copy import deepcopy
import traceback

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
            d1[k] = deepcopy(v)
        else:
            if type(v) == dict:
                d1[k] = accumulate_dicts(d1[k], v)
            else:
                d1[k] += v
    return d1

FIELD_GOAL_MADE = 1
FIELD_GOAL_MISSED = 2
FREE_THROW = 3
REBOUND = 4
TURNOVER = 5
FOUL = 6
EJECTION = 11
END_OF_PERIOD = 13


class Game():
    def __init__(self, game):
        self.game_id = game['game_id']
        self.date = game['date']
        self.home_team_id = game['home_team_id']
        self.home_team_name = team_ids[str(game['home_team_id'])]
        self.away_team_id = game['visitor_team_id']
        self.away_team_name = team_ids[str(game['visitor_team_id'])]
        self.plays = defaultdict(lambda: None)
        self.shots = defaultdict(lambda: None)
        self.bad_plays = 0
        self.player_stats = {}

        self.load_shots()
        self.load_plays()
        self.parse_plays()
    
    def load_shots(self):
        shotloader = StatsNbaShotsLoader(self.game_id, "file", "/data")
        for shot in shotloader.items:
            self.shots[shot.data['game_event_id']] = Shot(shot.data)

    def load_plays(self):
        playloader = StatsNbaPossessionLoader(self.game_id, "file", "/data")
        for play in playloader.items:
            p = Play(play, shots=self.shots)
            self.plays[p.play_id] = p

    def parse_plays(self):
        for play_id, play in self.plays.items():
            accumulate_dicts(self.player_stats, play.player_stats)

    def __repr__(self):
        small_dict = self.__dict__
        small_dict.pop('plays')
        small_dict.pop('shots')
        for team, player_stats in small_dict['player_stats'].items():
            for player, stats in player_stats.items():
                stats.pop('shots')

        return yaml.dump({self.game_id: small_dict}, allow_unicode=True, default_flow_style=False)
    
    def to_dict(self):
        plays = []
        for _, play in self.plays.items():
            play_dict = play.__dict__
            play_dict.pop('events')
            plays.append(play_dict)

        outdict = {
            'stats': self.player_stats,
            'plays': plays,
            'home': self.home_team_name,
            'away': self.away_team_name,
            'date': self.date,
            'game_id': self.game_id,
            'bad_plays': self.bad_plays
        }
        return outdict

class Play():
    def __init__(self, play, shots=defaultdict(lambda: None), players_on_court={}):
        self.play_id = str(play.data['period'] * 1000 + play.data['number'])
        self.start_time = play.start_time
        self.end_time = play.end_time
        self.offense_team = teamname(play.offense_team_id)
        self.possession_start_type = play.possession_start_type
        self.player_stats = {}
        self.events = []

        attempts = self.split_events(play.events)
        
        for attempt in attempts:
            if len(attempt) > 0:
                EventTree(attempt)

    def split_events(self, events):
        attempts = [[]]
        for event in events:
            attempts[-1].append(event)
            if (event.event_type == REBOUND 
                and event.is_real_rebound
                and event.oreb):
                attempts.append([])

            if (event.event_type == FREE_THROW
                and event.next_event is not None
                and event.next_event.event_type != FREE_THROW
                and event.foul_that_led_to_ft is not None
                and event.foul_that_led_to_ft.is_flagrant):
                attempts.append([])

        return attempts

    def __repr__(self):
        small_dict = self.__dict__
        small_dict.pop('events')
        return yaml.dump({self.play_id: small_dict}, allow_unicode=True, default_flow_style=False)

class EventTree():
    def __init__(self, events):
        self.offense_team = None
        self.offense_players = None
        self.defense_team = None
        self.defense_players = None

        self.n_shots = 0
        self.n_rebounds = 0
        self.n_free_throws = 0
        self.n_free_throws_made = 0
        self.last_free_throw_made = False
        self.shot_made = False

        self.shooter = None
        self.freethrower = None
        self.assister = None
        self.blocker = None

        self.fouler = None
        self.fouled = None
        self.offensive_fouler = None
        self.shooting_fouler = None
        self.foul_type = None

        self.shot_type = None
        self.shot_value = None
        self.shot_distance = None

        self.defensive_rebounder = None
        self.offensive_rebounder = None

        self.turnoverer = None
        self.turnover_type = None
        self.stealer = None

        self.second_chance = False
        print()
        for event in events:
            print(event.event_type, event.description)
            
        self.parse_event_contents(events)
        self.create_tree()

    def create_tree(self):
        tree = []
        
        if self.shot_value is None:
            self.shot_value = self.n_free_throws

        if self.shooter:
            tree += ["SHOT", self.shooter, self.shot_value, self.shot_type, self.shot_distance]
            tree += [self.assister, self.blocker]
            tree += ["MAKE" if self.shot_made else "MISS"]
            tree += [self.shooting_fouler, self.n_free_throws, self.n_free_throws_made]
            tree += [self.offensive_rebounder, self.defensive_rebounder]
        elif self.turnoverer:
            tree += ["TURNOVER ", self.turnoverer, self.stealer, self.offensive_fouler, self.fouled]
        elif self.n_free_throws > 0:
            tree += ["FOUL ", self.fouler, self.fouled, self.n_free_throws, self.n_free_throws_made]
            tree += [self.offensive_rebounder, self.defensive_rebounder]
        else:
            tree += ["UNKNOWN"]
        print(tree)

    def parse_event_contents(self, events):
        n_shots = 0 # Should always be 0 or 1
        shot_event = None
        contains_foul = False
        shot_made = None
        for event in events:
            if event.event_type == FIELD_GOAL_MADE:
                self.parse_shot(event)
            if event.event_type == FIELD_GOAL_MISSED:
                self.parse_shot(event)
            if event.event_type == FREE_THROW:
                self.parse_free_throw(event)
            if event.event_type == REBOUND:
                self.parse_rebound(event)
            if event.event_type == FOUL:
                self.parse_foul(event)
            if event.event_type == TURNOVER:
                self.parse_turnover(event)

    def parse_shot(self, event):
        self.parse_teams(event)
        self.n_shots += 1
        self.shooter = name(event.data['player1_id'])
        self.shot_distance = event.distance
        self.shot_type = event.shot_type
        self.shot_value = event.shot_value
        self.second_chance = event.is_second_chance_event()
        self.shot_made = (event.event_type == FIELD_GOAL_MADE)
        
        if event.is_assisted:
            self.assister = name(event.data['player2_id'])

        if event.is_blocked:
            self.blocker = name(event.data['player3_id'])

    def parse_free_throw(self, event):
        self.n_free_throws += 1
        self.n_free_throws_made += event.is_made
        if self.offense_team is None:
            self.parse_teams(event.event_for_efficiency_stats)
        if event.is_end_ft:
            self.last_free_throw_made = event.is_made
        self.freethrower = name(event.data['player1_id'])
    
    def parse_rebound(self, event):
        if event.is_real_rebound:
            self.n_rebounds += 1
            if event.oreb:
                self.offensive_rebounder = name(event.data['player1_id'])
            else:
                self.defensive_rebounder = name(event.data['player1_id'])
    
    def parse_foul(self, event):
        self.foul_type = event.foul_type_string
        self.fouler = name(event.data['player1_id'])
        if event.is_offensive_foul:
            self.offensive_fouler = self.fouler
        if 'player3_id' in event.data:
            self.fouled = name(event.data['player3_id'])
        if event.is_shooting_foul:
            self.shooting_fouler = self.fouler
            self.shooter = self.fouled
        
    def parse_turnover(self, event):
        self.parse_teams(event)
        self.turnoverer = name(event.data['player1_id'])
        if event.is_steal:
            self.stealer = name(event.data['player3_id'])
    
    def parse_teams(self, event):  
        self.offense_team = teamname(event.get_offense_team_id())

        for team, players in event.current_players.items():
            if teamname(team) is not self.offense_team:
                self.defense_team = teamname(team)
                self.defense_players = [name(player_id) for player_id in players]
            else:
                self.offense_players = [name(player_id) for player_id in players]
        
class Event():
    def __init__(self, event, shot=None):
        self.shot = shot
        self.player_stats = {}
        self.event_id = event.data['event_num']
        self.description = event.data['description']
        self.is_second_chance = event.is_second_chance_event()
        self.offense_team = teamname(event.get_offense_team_id())
        self.defense_team = None
        for team, players in event.current_players.items():
            team_name = teamname(team)
            if team_name is not self.offense_team:
                self.defense_team = team_name
            self.player_stats[team_name] = {name(player_id):{} for player_id in players}
        event_type_enum = event.data['event_type']
        self.event_type = str(event_type_enum)
        if event_type_enum == 1:
            self.event_type = 'field_goal_made'
            self.parse_field_goal(event)

        if event_type_enum == 2:  
            self.event_type = 'field_goal_missed'
            self.parse_field_goal(event)

        if event_type_enum == 3:  
            self.event_type = 'free_throw'
            shooter = name(event.data['player1_id'])
            self.player_stats[self.offense_team][shooter]['fta'] = 1
            if event.is_made:
                self.player_stats[self.offense_team][shooter]['ftm'] = 1
                self.player_stats[self.offense_team][shooter]['pts'] = 1

        if event_type_enum == 4:  
            self.event_type = 'rebound'
            rebounder = name(event.data['player1_id'])
            if rebounder is not None:
                if rebounder in self.player_stats[self.defense_team]:
                    self.player_stats[self.defense_team][rebounder]['drb'] = 1
                else:
                    self.player_stats[self.offense_team][rebounder]['orb'] = 1

        if event_type_enum == 5:  
            self.event_type = 'turnover'
            turnoverer = name(event.data['player1_id'])
            if turnoverer in self.player_stats[self.offense_team]:
                self.player_stats[self.offense_team][turnoverer]['tov'] = 1
            if 'player3_id' in event.data:
                stealer = name(event.data['player3_id'])
                self.player_stats[self.defense_team][stealer]['stl'] = 1

        if event_type_enum == 6:  
            self.event_type = 'foul'
            if event.counts_as_personal_foul:
                fouler = name(event.data['player1_id'])
                if fouler in self.player_stats[self.defense_team]:
                    self.foul_type = 'defensive'
                    self.player_stats[self.defense_team][fouler]['d_foul'] = 1
                else:
                    self.foul_type = 'offensive'
                    self.player_stats[self.offense_team][fouler]['o_foul'] = 1

            if 'player3_id' in event.data:
                foulee = name(event.data['player3_id'])
                if foulee in self.player_stats[self.defense_team]:
                    self.player_stats[self.defense_team][foulee]['d_foul_drawn'] = 1
                else:
                    self.player_stats[self.offense_team][foulee]['o_foul_drawn'] = 1

        if event_type_enum == 7:  
            self.event_type = 'violation'
            self.violator = name(event.data['player1_id'])
            print(self.violator)

        if event_type_enum == 8:  
            self.event_type = 'substitution'
            #self.sub_in = name(event.data['player1_id'])
            #self.sub_out = name(event.data['player2_id']) 

        if event_type_enum == 9: 
            self.event_type = 'timeout'

        if event_type_enum == 10: 
            self.event_type = 'jump_ball'
            #self.jump_winner = name(event.data['player1_id'])
            #self.tipped_to = name(event.data['player1_id'])
            #self.jump_loser = name(event.data['player3_id'])
        
        if event_type_enum == 11: 
            self.event_type = 'ejection'
            #self.ejected = name(event.data['player1_id'])
        
        if event_type_enum == 12: 
            self.event_type = 'period_start'
            #self.starters = event.data['period_starters']
            
        if event_type_enum == 13: 
            self.event_type = 'period_end'
        


    def parse_field_goal(self, event):
        shooter = name(event.data['player1_id'])
        self.player_stats[self.offense_team][shooter]['fga'] = 1
        shot_type_str = '3p' if event.shot_value == 3 else '2p'
        self.player_stats[self.offense_team][shooter][shot_type_str + 'a'] = 1

        if event.is_made:
            self.player_stats[self.offense_team][shooter]['fgm'] = 1
            self.player_stats[self.offense_team][shooter][shot_type_str + 'm'] = 1
            self.player_stats[self.offense_team][shooter]['pts'] = event.shot_value

        shot_stats = self.shot.shot_stats
        shot_stats['made'] = event.is_made
        shot_stats['2nd_chance'] = event.is_second_chance_event()
        shot_stats['value'] = event.shot_value
        shot_stats['assisted'] = event.is_assisted
        self.player_stats[self.offense_team][shooter]['shots'] = [shot_stats]

        if event.is_assisted:
            assister = name(event.data['player2_id'])
            if assister in self.player_stats[self.offense_team]:
                self.player_stats[self.offense_team][assister]['ast'] = 1

        if event.is_blocked:
            blocker = name(event.data['player3_id'])
            if blocker in self.player_stats[self.defense_team]:
                self.player_stats[self.defense_team][blocker]['blk'] = 1

    def __repr__(self):
        return yaml.dump({self.event_id: self.__dict__}, allow_unicode=True, default_flow_style=False)

class Shot():
    def __init__(self, shot):
        self.event_id = shot['game_event_id'] 
        self.player = name(shot['player_id'])
        self.team = teamname(shot['team_id'])
        self.period = shot['period']
        self.minutes_remaining = shot['minutes_remaining'] + shot['seconds_remaining'] / 60
        self.type = shot['shot_type']
        self.type_detailed = shot['action_type']
        self.event_type = shot['event_type']
        self.zone = shot['shot_zone_basic']
        self.area = shot['shot_zone_area']
        self.distance = shot['shot_distance']
        self.x = shot['loc_x'] / 10 # to feet
        self.y = shot['loc_y'] / 10 # to feet
        self.attempted = shot['shot_attempted_flag']
        self.made = shot['shot_made_flag']
        self.shot_stats = {
            'x': self.x,
            'y': self.y,
            'd': self.distance,
            'description': self.type_detailed
        }

    def __repr__(self):
        return yaml.dump({self.event_id: self.__dict__}, allow_unicode=True, default_flow_style=False)

season_types = ["Regular Season", "Playoffs"]
seasons = [str(i) + "-" + str(i+1)[-2:] for i in range(2018, 2019)]

all_games = []
for season in seasons:
    for season_type in season_types:
        season_games = StatsNbaGameFinderLoader("nba", season, season_type, "file", "/data")
        for game in season_games.items:
            all_games.append(game.data)

i = 0
for game in all_games:
    i = i + 1
    print(i)
    g = Game(game)
print("end")
# total_bad_plays = 0
# total_errs = 0
# for i, game in enumerate(all_games):
#     g = Game(game)
#     try:
#         with open(os.path.join('.', 'games', g.game_id + '.json'), 'w') as f:
#             json.dump(g.to_dict(), f)
#         total_bad_plays += g.bad_plays
#         print(i, "-", g.bad_plays, '-', total_bad_plays)
#     except Exception as e:
#         traceback.print_exc()
#         total_errs += 1
#         print(game, total_errs)
#     break
# print(g)
# print(g.plays[0])