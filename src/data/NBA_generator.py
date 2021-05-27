
def special_generator():
    from src.loader_pipeline import open_db
    from src.data.play import Play
    from src.data.game import Game
    from src.data.possession import Possession

    with open_db() as db:
        games = db["raw_data/2001_playoffs/games"]
        possessions = db["raw_data/2001_playoffs/possessions"]
        plays = db["raw_data/2001_playoffs/plays"]
        season_info = db["raw_data/2001_playoffs/season_info"]
        for game_data in games:
            game = Game(game_data)
            for possession_data in possessions[game.possession_start_idx:game.possession_end_idx]:
                possession = Possession(possession_data)
                for play_data in plays[possession.play_start_idx:possession.play_end_idx]:
                    play = Play(play_data)
                    yield game, possession, play



if __name__ == "__main__":
    print("123 testing")