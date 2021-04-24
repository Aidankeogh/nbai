from src.data.box_stats import box_stats
from src.utilities.global_timers import timeit

@timeit
def parse_box_stats(play, rosters):
    offense_roster = rosters[play.offense_team]
    defense_roster = rosters[play.defense_team]

    o_stats = box_stats(offense_roster)
    d_stats = box_stats(defense_roster)

    for player in play.offense_roster:
        o_stats[player, 'o_pos'] = 1 - play.is_second_chance
        o_stats[player, 'o+-'] = play.score_change

    for player in play.defense_roster:
        d_stats[player, 'd_pos'] = 1 - play.is_second_chance
        d_stats[player, 'd+-'] = -play.score_change

    if play.shooter != "none":
        o_stats[play.shooter, '2pa'] = not play.is_3
        o_stats[play.shooter, '2pm'] = play.shot_made and not play.is_3
        o_stats[play.shooter, '3pa'] = play.is_3
        o_stats[play.shooter, '3pm'] = play.shot_made and play.is_3
        o_stats[play.shooter, 'pts'] += play.shot_made * (2 + play.is_3)

    if play.free_thrower != "none" and play.free_thrower in o_stats.player_idxes:
        #o_stats[play.o_ft, 'fta'] = play.ft_made + play.ft_missed
        n_free_throws = (play.initial_event == "foul_over_limit") * 2
        if play.shot_made and play.shot_fouled:
            n_free_throws = 1
        elif play.shot_fouled:
            n_free_throws = 2 + play.is_3
        free_throws_made = (
            play.first_free_throw_made + 
            play.middle_free_throw_made +
            play.last_free_throw_made
        )
        o_stats[play.free_thrower, 'ftm'] = free_throws_made
        o_stats[play.free_thrower, 'fta'] = n_free_throws
        o_stats[play.free_thrower, 'pts'] += free_throws_made

    if play.defensive_rebounder != "none":
        d_stats[play.defensive_rebounder, 'drb'] = 1

    if play.offensive_rebounder != "none":
        o_stats[play.offensive_rebounder, 'orb'] = 1

    if play.assister != "none":
        o_stats[play.assister, 'ast'] = 1
    if play.stealer != "none":
        d_stats[play.stealer, 'stl'] = 1
    if play.blocker != "none":
        d_stats[play.blocker, 'blk'] = 1
    if play.turnoverer != "none":
        o_stats[play.turnoverer, 'tov'] = 1

    if play.over_limit_fouler != "none" and play.over_limit_fouler in d_stats.player_idxes:
        d_stats[play.over_limit_fouler, 'pfs'] = 1
    if play.shooting_fouler != "none" and play.shooting_fouler in d_stats.player_idxes:
        d_stats[play.shooting_fouler, 'pfs'] = 1
    if play.common_fouler != "none" and play.common_fouler in d_stats.player_idxes:
        d_stats[play.common_fouler, 'pfs'] = 1
    
    return o_stats, d_stats