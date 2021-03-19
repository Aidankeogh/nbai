import msgpack
import torch
import os
from torch.utils.data import Dataset

class NbaPosessionDataset(Dataset):
    def __init__(self, db=None, years=None, prebuilt_path=None):

        if prebuilt_path is not None:
            with open(prebuilt_path, "rb") as f:
                self.data = msgpack.loads(f.read())
            self.year_ids = self.data['year_ids']
            self.item_keys = self.data['item_keys']
        else:
            self.year_ids = [str(y-1) + "-" + str(y)[-2:] for y in years]

            self.item_keys = set()
            for year_id in self.year_ids:
                keys = db['item_keys_{}'.format(year_id)]
                if keys is not None:
                    full_item_keys = msgpack.loads(keys)
                    for i in full_item_keys:
                        self.item_keys.add( (i[1], i[2]) )
            self.item_keys = list(self.item_keys)
            
            self.data = {}
            for i, (game_key, pos_key) in enumerate(self.item_keys):
                if i % 100000 == 0:
                    print(f"Building dataset, {i * 100.0 / len(self.item_keys)}% complete")
        
                if game_key not in self.data:
                    self.data[game_key] = msgpack.loads(db[game_key])
                if pos_key not in self.data:
                    self.data[pos_key] = msgpack.loads(db[pos_key])
            
            self.data['item_keys'] = self.item_keys
            
            self.data['year_ids'] = self.year_ids

    def __len__(self):
        return len(self.item_keys)

    def __getitem__(self, idx):
        game, possession = self.item_keys[idx]
        game = self.data[game]
        pos = self.data[possession]
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
    db.set_namespace(config['loader']['key'])
    dset = NbaPosessionDataset(db=db, years=years)

    if not os.path.exists('prebuilt_datasets'):
        os.mkdir('prebuilt_datasets')

    save_path = f"prebuilt_datasets/{config['dataset']['key']}.msgpack"
    with open(save_path, "wb") as f:
        msgpack.dump(dset.data, f)

    db.set_namespace(config['dataset']['key'])
    db['save_path'] = save_path.encode()
    db['completed'] = 'true'.encode()
