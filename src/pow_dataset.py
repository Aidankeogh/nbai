import msgpack
import torch
from torch.utils.data import Dataset

class NbaPosessionDataset(Dataset):
    def __init__(self, db, years):
        self.year_ids = [str(y-1) + "-" + str(y)[-2:] for y in years]
        self.item_keys = []
        self.data = {}
        for year_id in self.year_ids:
            keys = db['item_keys_{}'.format(year_id)]
            if keys is not None:
                full_item_keys = msgpack.loads(keys)
                game_and_possession_keys = [ [i[1], i[2]] for i in full_item_keys] 
                self.item_keys += game_and_possession_keys
        
        for i, (game_key, pos_key) in enumerate(self.item_keys):
            if i % 1000 == 0:
                print(f"Building dataset, {i * 100.0 / len(self.item_keys)}% complete")
    
            if game_key not in self.data:
                self.data[game_key] = msgpack.loads(db[game_key])
            if pos_key not in self.data:
                self.data[pos_key] = msgpack.loads(db[pos_key])
        
        with open("pow_dataset.msgpack") as f:
            self.data['item_keys'] = self.item_keys
            msgpack.dump(self.data, f)

    def __getitem__(self, idx):
        game, possession = self.item_keys[idx]
        game = msgpack.loads(self.db.get(game))
        pos = msgpack.loads(self.db.get(possession))
        start_tensor = torch.tensor([
            pos['scores'][pos['offense_team']] - pos['score_change'],
            pos['scores'][pos['defense_team']],
            pos['penalty_fouls'][pos['offense_team']],
            pos['penalty_fouls'][pos['defense_team']] - pos['foul_change'],
            pos['start_time'],
            1 # has posession indicator
        ]).float()

        end_tensor = torch.tensor([
            pos['scores'][pos['offense_team']],
            pos['scores'][pos['defense_team']],
            pos['penalty_fouls'][pos['offense_team']],
            pos['penalty_fouls'][pos['defense_team']],
            pos['end_time'],
            0
        ]).float()

        off_score = game['scores'][pos['offense_team']] if pos['offense_team'] in game['scores'] else 0
        def_score = game['scores'][pos['defense_team']] if pos['defense_team'] in game['scores'] else 0
        fin_tensor = torch.Tensor([
            off_score > def_score
        ]).float()

        return start_tensor, end_tensor, fin_tensor

def create_dataset(config, db, years):
    return NbaPosessionDataset(db, years)
