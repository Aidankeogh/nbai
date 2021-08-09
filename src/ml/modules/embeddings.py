from os import remove
import torch.nn as nn
from src.utilities.embedding_utilities import n_players

class PlayerEmbeddings(nn.Module):
    def __init__(
            self,
            embedding_dim=32,
            model_dim=512,
            n_teams=32,
            n_players=n_players,
            remove_team=False,
            upsample_embeddings=True,
            **kwargs
        ):
        super().__init__()
        self.remove_team = remove_team
        self.upsample = upsample_embeddings
        self.team_embeddings = nn.Embedding(n_teams, embedding_dim)
        self.player_embeddings = nn.Embedding(n_players, embedding_dim)
        if self.upsample:
            self.team_offense_upsample = nn.Sequential(
                nn.Conv1d(embedding_dim, model_dim, 1),
                nn.BatchNorm1d(model_dim)
            )
            self.team_defense_upsample = nn.Sequential(
                nn.Conv1d(embedding_dim, model_dim, 1),
                nn.BatchNorm1d(model_dim)
            )
            self.player_offense_upsample = nn.Sequential(
                nn.Conv1d(embedding_dim, model_dim, 1),
                nn.BatchNorm1d(model_dim)
            )
            self.player_defense_upsample = nn.Sequential(
                nn.Conv1d(embedding_dim, model_dim, 1),
                nn.BatchNorm1d(model_dim)
            )
        elif model_dim != embedding_dim:
            raise f"Model and embedding dim mismatch {model_dim} vs {embedding_dim}! Turn on upsample_embeddings=True"

    def forward(
        self, play
    ):
        offense_roster = play["offense_roster"]
        defense_roster = play["defense_roster"]
        offense_team = play["offense_team"]
        defense_team = play["defense_team"]

        # Removes team info, but leaves them as a single learnable param for each team
        if self.remove_team:
            offense_team = offense_team * 0
            defense_team = defense_team * 0

        offense_roster = self.player_embeddings(offense_roster)
        defense_roster = self.player_embeddings(defense_roster)
        offense_team = self.team_embeddings(offense_team)
        defense_team = self.team_embeddings(defense_team)

        if self.upsample:
            offense_roster = self.player_offense_upsample(offense_roster.swapaxes(1,2)).swapaxes(1,2)
            defense_roster = self.player_defense_upsample(defense_roster.swapaxes(1,2)).swapaxes(1,2)
            offense_team = self.team_offense_upsample(offense_team.swapaxes(1,2)).swapaxes(1,2)
            defense_team = self.team_defense_upsample(defense_team.swapaxes(1,2)).swapaxes(1,2)
    
        return offense_roster, defense_roster, offense_team, defense_team
