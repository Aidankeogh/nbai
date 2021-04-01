from src.data_format import Play
from src.data_utils import player_name, player_id, team_name, team_id

FIELD_GOAL_MADE = 1
FIELD_GOAL_MISSED = 2
FREE_THROW = 3
REBOUND = 4
TURNOVER = 5
FOUL = 6
EJECTION = 11
END_OF_PERIOD = 13

def parse_play(in_data, out_data):
    index, events = in_data['play']
    play = Play()
    play.is_second_chance = index > 1
    for event in events:
        if event.event_type == FIELD_GOAL_MADE:
            parse_shot(play, event)
        if event.event_type == FIELD_GOAL_MISSED:
            parse_shot(play, event)
        if event.event_type == FREE_THROW:
            parse_free_throw(play, event)
        if event.event_type == REBOUND:
            parse_rebound(play, event)
        if event.event_type == FOUL:
            parse_foul(play, event)
        if event.event_type == TURNOVER:
            parse_turnover(play, event)
    out_data['plays'].append(play.data)

def parse_shot(play, event):
    play.fga = 1
    play.o_fga = event.data['player1_id']
    play.shot_dist = event.distance if event.distance is not None else -1
    shot_type = event.shot_type
    #print(shot_type)
    play.is3 = (event.shot_value == 3)
    play.is2 = (event.shot_value == 2)
    play.make = (event.event_type == FIELD_GOAL_MADE)
    play.miss = (event.event_type != FIELD_GOAL_MADE)
    
    if event.is_assisted:
        play.assist = 1
        play.o_assist = event.data['player2_id']

    if event.is_blocked:
        play.block = 1
        play.d_block = event.data['player3_id']

def parse_free_throw(play, event):
    parse_teams(play, event.event_for_efficiency_stats)
    play.ft_made += event.is_made
    play.ft_missed += 1 - event.is_made
    play.o_ft = event.data['player1_id']

def parse_rebound(play, event):
    if event.is_real_rebound:
        if event.oreb:
            play.oreb = 1
            play.o_oreb = event.data['player1_id']
        else:
            play.dreb = 1
            play.d_dreb = event.data['player1_id']

def parse_foul(play, event):
    foul_type = event.foul_type_string
    fouler = event.data['player1_id']
  
    fouled = event.data['player3_id'] if 'player3_id' in event.data else 0
    if event.is_offensive_foul:
        play.o_foul_off = fouler
        play.d_foul_off = fouled
    elif event.is_shooting_foul:
        play.d_foul_shot = fouler
        play.o_foul_shot = fouled
        play.o_fga = fouled
    elif event.number_of_fta_for_foul:
        play.d_foul_over_limit = fouler
        play.o_foul_over_limit = fouled
    
def parse_turnover(play, event):
    parse_teams(play, event)
    play.o_tov = event.data['player1_id']
    if event.is_steal:
        play.d_steal = event.data['player3_id']

def parse_teams(play, event):  
    off_team_id = event.get_offense_team_id()
    play.off_team_name = team_name(off_team_id)
    for team, players in event.current_players.items():
        if team != off_team_id:
            play.def_team_name = team_name(team)
            play.def_team = players
        else:
            play.off_team = players

def split_events(events):
    events = [e for e in events if e.event_type not in [9, 10, 12, 13, 18]]
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

    if len(attempts[0]) == 0:
        attempts = []
    return enumerate(attempts)

