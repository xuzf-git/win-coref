""" Describes AnaphicityScorer, a torch module that for a matrix of
mentions produces their anaphoricity scores.
"""
import torch

from coref import utils
from coref.config import Config


class AnaphoricityScorer(torch.nn.Module):
    """ Calculates anaphoricity scores by passing the inputs into a FFNN """

    def __init__(self, in_features: int, config: Config):
        super().__init__()
        hidden_size = config.hidden_size
        if not config.n_hidden_layers:
            hidden_size = in_features
        layers = []
        for i in range(config.n_hidden_layers):
            layers.extend([
                torch.nn.Linear(hidden_size if i else in_features, hidden_size),
                torch.nn.LeakyReLU(),
                torch.nn.Dropout(config.dropout_rate)
            ])
        self.hidden = torch.nn.Sequential(*layers)
        self.out = torch.nn.Linear(hidden_size, out_features=1)

    def forward(
        self,
        pairs_batch: torch.Tensor,
    ) -> torch.Tensor:
        """ scores the pairs and returns the scores.

        Args:
            pairs_batch (torch.Tensor): [n_pairs, features_emb]

        Returns:
            torch.Tensor [n_pairs, n_ants + 1] anaphoricity scores for the pairs
        """
        # [n_pairs, features_emb]
        scores = self._ffnn(pairs_batch)
        # scores = utils.add_dummy(scores, eps=True)

        return scores

    def _ffnn(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates anaphoricity scores.

        Args:
            x: tensor of shape [batch_size, n_ants, n_features]

        Returns:
            tensor of shape [batch_size, n_ants]
        """
        x = self.out(self.hidden(x))
        return x.squeeze(1)