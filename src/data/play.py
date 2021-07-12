from src.thought_path import DataConfig, ThoughtPath
from src.utilities.global_timers import timeit
import torch
from src.data.data_utils import (
    player_name,
    team_name,
    FOUL,
    FIELD_GOAL_MADE,
    FIELD_GOAL_MISSED,
    FREE_THROW,
    REBOUND,
    TURNOVER,
    FOUL,
    EJECTION,
    END_OF_PERIOD,
)

play_config = DataConfig("src/data/play.yaml")
data_keys = play_config.data_keys
slice_keys = play_config.slice_keys


class Play(ThoughtPath):
    def __init__(self, data=None):
        super().__init__(play_config, data=data)

    def __getattr__(self, key):
        if key == "score_change":
            return self.shot_made * (2 + self.is_3) + self.free_throws_made
        else:
            return super().__getattr__(key)

    def __repr__(self) -> str:
        out_str = ""
        if self.shooter:
            out_str += f"{self.shooter} "
            out_str += "made " if self.shot_made else "missed "
            out_str += "3 point " if self.is_3 else "2 point "
            out_str += f"{self.shot_type} "  # TODO: Fix shot distance
            if self.shooting_fouler:
                out_str += f"fouled by {self.shooting_fouler} "
            if self.assister:
                out_str += f"assisted by {self.assister} "
            if self.blocker:
                out_str += f"blocked by {self.blocker} "
        if self.over_limit_fouler:
            out_str += (
                f"| {self.over_limit_foul_drawer} fouled by {self.over_limit_fouler} "
            )
        if self.free_thrower:
            fts_made = self.free_throws_made
            out_str += f"| {self.free_thrower} made {fts_made} fts "
        if self.turnoverer:
            out_str += f"| Turnover by {self.turnoverer} "
            if self.stealer:
                out_str += f" stolen by {self.stealer} "
        if self.offensive_fouler:
            out_str += f"| Offensive foul by {self.offensive_fouler} "
        if self.offensive_rebounder:
            out_str += f"| Offensive rebound by {self.offensive_rebounder} "
        if self.defensive_rebounder:
            out_str += f"| Defensive rebound by {self.defensive_rebounder} "

        return out_str


@timeit
def parse_play(in_data, out_data):
    index, events = in_data["play"]
    if len(events) == 0:
        return
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
    if play.offense_team is None or play.defense_team is None:
        return
        # raise Exception("ERROR")
    out_data["plays"].append(play.data)


def parse_shot(play, event):
    parse_teams(play, event)
    play.initial_event = "shot"
    play.shooter = player_name(event.data["player1_id"])
    play.shot_distance = event.distance if event.distance is not None else -1
    play.shot_type = event.shot_type
    play.is_3 = event.shot_value == 3
    play.shot_made = event.event_type == FIELD_GOAL_MADE

    if event.is_assisted:
        play.assisted = 1
        play.assister = player_name(event.data["player2_id"])

    if event.is_blocked:
        play.blocked = 1
        play.blocker = player_name(event.data["player3_id"])


def parse_free_throw(play, event):
    if "Technical" in event.description or "Clear Path" in event.description:
        return
    if "of" not in event.description:
        print(event.description)
        return

    if (
        event.foul_that_led_to_ft is not None
    ):  ## TODO: Figure out how the hell to correctly count flagrants
        if not event.foul_that_led_to_ft.is_shooting_foul:
            play.initial_event = "foul_over_limit"
        parse_teams(play, event.event_for_efficiency_stats)

    desc = event.description.split(" ")
    idx = desc.index("of")
    n_free_throws = int(desc[idx + 1])
    curr_free_throw = int(desc[idx - 1])
    if curr_free_throw == n_free_throws:
        play.last_free_throw_made = event.is_made
    play.free_throws_made += event.is_made
    play.free_throws_attempted += 1
    play.free_thrower = player_name(event.data["player1_id"])


def parse_rebound(play, event):
    if event.is_real_rebound:
        if event.oreb:
            play.rebound_type = "offensive"
            play.offensive_rebounder = player_name(event.data["player1_id"])
        else:
            play.rebound_type = "defensive"
            play.defensive_rebounder = player_name(event.data["player1_id"])


def parse_foul(play, event):
    fouled = player_name(event.data["player3_id"] if "player3_id" in event.data else 0)
    fouler = player_name(event.data["player1_id"] if "player1_id" in event.data else 0)

    if event.is_offensive_foul or event.data["event_action_type"] == 26:
        play.offensive_foul = True
        play.offensive_fouler = fouler
        play.offensive_foul_drawer = fouled
    elif event.is_shooting_foul:
        parse_teams(play, event)
        play.initial_event = "shot"
        play.shot_fouled = 1
        play.shooting_fouler = fouler
        # if the player isn't there (substitution) keep walking back until they are there
        prev_event = event.previous_event
        while (
            play.shooting_fouler not in play.defense_roster
            and prev_event.previous_event is not None
        ):
            parse_teams(play, prev_event)
            prev_event = prev_event.previous_event

    elif event.number_of_fta_for_foul:
        parse_teams(play, event)
        play.initial_event = "foul_over_limit"
        play.over_limit_fouler = fouler
        play.over_limit_foul_drawer = fouled
        if play.over_limit_fouler not in play.defense_roster:
            return
    else:
        parse_teams(play, event)
        play.common_fouler = fouler
        play.common_foul_drawer = fouled
        if play.common_fouler not in play.defense_roster:
            return


def parse_turnover(play, event):
    parse_teams(play, event)
    play.initial_event = "turnover"
    play.turnoverer = player_name(event.data["player1_id"])
    if event.is_steal:
        play.stolen = 1
        play.stealer = player_name(event.data["player3_id"])


def parse_teams(play, event):
    off_team_id = event.get_offense_team_id()
    play.offense_team = team_name(off_team_id)
    for team, players in event.current_players.items():
        if team != off_team_id:
            play.defense_team = team_name(team)
            play.defense_roster = [player_name(p) for p in players]
        else:
            play.offense_roster = [player_name(p) for p in players]


def split_events(events):
    events = [e for e in events if e.event_type not in [9, 10, 12, 13, 18]]
    attempts = [[]]
    for event in events:
        attempts[-1].append(event)
        if event.event_type == REBOUND and event.is_real_rebound and event.oreb:
            attempts.append([])

        if (
            event.event_type == FREE_THROW
            and event.next_event is not None
            and event.next_event.event_type != FREE_THROW
            and event.foul_that_led_to_ft is not None
            and event.foul_that_led_to_ft.is_flagrant
        ):
            attempts.append([])

    if len(attempts[0]) == 0:
        attempts = []
    return enumerate(attempts)


if __name__ == "__main__":
    p = Play()
    curry = "stephen-curry"
    lebron = "lebron-james"
    kd = "kevin-durant"
    p.initial_event = "shot"
    p.shooter = curry
    p.offense_roster = [curry, curry, kd, kd, kd]
    p.defense_roster = [lebron, lebron, lebron, lebron, lebron]
    shot_probs = [0.5, 0.3, 0.1, 0.1, 0]
    p.shot_type = shot_probs
    assert p.offense_roster[1] == "stephen-curry"
    assert p.shooter == "stephen-curry"
    assert p.shot_type == "Arc3"
    assert list(p["shot_type"]) == shot_probs
    print(p.offense_roster)
    print(p.shot_type_longmidrange)
    print(p)
