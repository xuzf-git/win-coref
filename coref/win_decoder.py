""" Describes WordEncoder. Predict the top-k antecedent for the mentions
    by windows-pruning HOI method.
"""

from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import toml

from coref.config import Config
from coref.anaphoricity_scorer import AnaphoricityScorer
from coref import utils
from coref.const import Doc
from coref.pairwise_encoder import PairwiseEncoder

import argparse


class WinDecoder(torch.nn.Module):
    """Decode the antecedent sequence by windows-pruning HOI method."""

    def __init__(self, features: int, config: Config):
        """
        Args:
            features (int): the num of featues of the score input embeddings
            config (Config): the configuration of the current session
        """
        super().__init__()
        self.topk = config.top_k
        self.device = config.device
        self.win_builder = WinBuilder(features, config)

    def forward(
            self,  # type: ignore  # pylint: disable=arguments-differ  #35566 in pytorch
            mentions: torch.Tensor,
            doc: Doc) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns top-k antecedent sequence and scores for input mentions

        Args:
            mentions: a tensor of shape [n_mentions, features], the embedding of mentions.
            doc: a dictionary with the document data.

        Returns:
            FloatTensor of shape [k, n_mentions], local scores corresponds to top-k global score
            LongTensor of shape [k, n_mentions], antecedent indices corresponds to top-k global score
        """

        n_mentions = mentions.size(0)
        # generate the local structure and local score
        global_struct = torch.zeros(self.topk, n_mentions + 1).to(self.device)
        global_scores = torch.zeros(self.topk, n_mentions + 1).to(self.device)
        for windows_size in range(1, n_mentions):
            # build local_structure(d + 1) from global_structure(d)
            local_struct, local_scores = self.win_builder(mentions, doc, global_struct, global_scores, windows_size)

            # BeamSearch the top-k global structure

        return


class WinBuilder(torch.nn.Module):
    """
    Build the local structure and local scores for a sentence
    according to the topk global structure and global score
    """

    def __init__(self, in_features: int, config: Config) -> None:
        """
        Args:
            batch_size: int, the number of windows structure generated once time
        """
        super().__init__()
        self.batch_size = config.batch_size
        self.device = config.device
        self.pairwise_encoder = PairwiseEncoder(config)
        self.scorer = AnaphoricityScorer(in_features, config)

    @staticmethod
    def _get_features(mention_features: torch.Tensor, pair_features: torch.Tensor):
        """
        Args:
            mention_features: a tensor of shape [n_pairs, 2, mention_emb], the embedding of mentions.
            pair_features: a tensor of shape [n_pairs, pair_emb], the embedding of pairs.
        Returns:
            features: a tensor of shape [n_pairs, 3 * mention_emb + pair_emb], the embedding of pairs.
        """
        a_mentions = mention_features[:, 0, :].squeeze(1)
        b_mentions = mention_features[:, 1, :].squeeze(1)
        similarity = a_mentions * b_mentions
        features = torch.cat((a_mentions, b_mentions, similarity, pair_features), dim=1)
        return features

    def forward(self, mentions: torch.Tensor, doc: Doc, global_struct: torch.Tensor, global_scores: torch.Tensor,
                windows_size: int) -> torch.Tensor:
        """
        Args:
            mentions: a tensor of shape [n_mentions + 1, features], the embedding of mentions.
            doc: a dictionary with the document data.
            global_struct: a tensor of shape [k, n_mentions + 1], the top-k global structure
            global_scores: a tensor of shape [k, n_mentions + 1], the top-k global score
            windows_size: int, the current windows size
        
        Returns:
            local_struct: a tensor of shape [k, n_mentions + 1], the local structure
            local_scores: a tensor of shape [k, n_mentions + 1], the local score
        """
        local_struct_lst: List[torch.Tensor] = []
        local_scores_lst: List[torch.Tensor] = []
        n_windows = mentions.size(0) - windows_size
        # build local structure in-batch [batch_size, k, windows_size]
        for i in range(1, n_windows + 1, self.batch_size):
            global_struct_batch = global_struct[:, i:i + self.batch_size + windows_size]  # [k, b_s+w_s]
            global_scores_batch = global_scores[:, i:i + self.batch_size + windows_size]  # [k, b_s+w_s]
            real_batch_size = global_struct_batch.size(1) - windows_size
            # To generate a 3D tensor from a 2D tensor by sampling in a sliding window fashion [k, b_s, w_s+1]
            local_struct_batch = global_struct_batch.unfold(dimension=1, size=windows_size + 1, step=1)
            local_scores_batch = global_scores_batch[:, -real_batch_size:]  # [k, b_s]
            # new local structure: link the last mention of windows to the first mention, 0 means link to dummy
            new_link = torch.arange(i, i + real_batch_size).to(self.device).view(1, -1, 1).repeat(local_struct_batch.size(0), 1, 1)
            local_struct_batch_append = torch.cat((local_struct_batch[:, :, :windows_size], new_link), dim=2)
            # score the new local structure
            new_link_pairs = (torch.arange(i, i + real_batch_size).unsqueeze(1).to(self.device),
                              torch.arange(i, i + real_batch_size).unsqueeze(1).to(self.device) + windows_size)
            new_link_pairs = torch.cat(new_link_pairs, dim=1) - 1
            pairwise_features = self.pairwise_encoder(new_link_pairs, doc)
            features_batch = self._get_features(mentions[new_link_pairs], pairwise_features)
            new_link_pairs_scores = self.scorer(features_batch)
            local_scores_batch_append = new_link_pairs_scores.unsqueeze(0).repeat(local_scores_batch.size(0), 1)
            
            local_struct_lst.append(torch.cat((local_struct_batch, local_struct_batch_append), dim=0))
            local_scores_lst.append(torch.cat((local_scores_batch, local_scores_batch_append), dim=0))
        
        local_struct = torch.cat(local_struct_lst, dim=1).view(n_windows, -1, windows_size + 1)  # [n_windows, 2k, windows_size + 1]
        local_scores = torch.cat(local_scores_lst, dim=1).view(n_windows, -1)  # [n_windows, 2k]

        return local_struct, local_scores


def load_config(config_path: str, section: str) -> Config:
    config = toml.load(config_path)
    default_section = config["DEFAULT"]
    current_section = config[section]
    unknown_keys = (set(current_section.keys()) - set(default_section.keys()))
    if unknown_keys:
        raise ValueError(f"Unexpected config keys: {unknown_keys}")
    return Config(section, **{**default_section, **current_section})


if __name__ == '__main__':
    config = load_config(config_path="config.toml", section="DEFAULT")

    # model = WinDecoder(1024, 5)
    # a = model(torch.randn(12, 1024))
    model = WinBuilder(config)
    a = model(torch.randint(low=0, high=10, size=(5, 15)), torch.randn(5, 15), 6)