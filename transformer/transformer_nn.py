from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from gymnasium import spaces
import torch as th
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy


class TransformerNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
            self,
            feature_dim: int,
            last_layer_dim_pi: int = 48,
            last_layer_dim_vf: int = 48,
            num_offense: int = 3,
    ):
        super().__init__()

        self.emb_dim = feature_dim // num_offense

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        d_model = 64
        self.shared_net = nn.Sequential(
            nn.Linear(self.emb_dim, d_model),
            *[nn.TransformerEncoderLayer(d_model=d_model, nhead=1, dim_feedforward=64, dropout=0.1, layer_norm_eps=1e-05, batch_first=True, norm_first=False,
                                         bias=True) for _ in range(2)],
        )

        self.policy_net = nn.Sequential(
            *[nn.TransformerEncoderLayer(d_model=d_model, nhead=1, dim_feedforward=64, dropout=0.1, layer_norm_eps=1e-05, batch_first=True, norm_first=False,
                                         bias=True) for _ in range(1)],
            nn.Linear(d_model, last_layer_dim_pi // num_offense)
        )

        # Value network
        self.value_net = nn.Sequential(
            *[nn.TransformerEncoderLayer(d_model=d_model, nhead=1, dim_feedforward=64, dropout=0.1, layer_norm_eps=1e-05, batch_first=True, norm_first=False,
                                         bias=True) for _ in range(1)],
            nn.Linear(d_model, last_layer_dim_pi // num_offense)
        )

        self.num_offense = num_offense

    def get_shared(self, features):
        b, tc = features.shape
        features = features.view(b, self.num_offense, self.emb_dim)
        shared_features = self.shared_net(features)
        return shared_features

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        shared_features = self.get_shared(features)
        res = self.forward_actor(shared_features, shared=True), self.forward_critic(shared_features, shared=True)
        return res

    def forward_actor(self, features: th.Tensor, shared=False) -> th.Tensor:
        if not shared:
            features = self.get_shared(features)
        res = self.policy_net(features)
        res = res.view(features.size(0), -1)
        return res

    def forward_critic(self, features: th.Tensor, shared=False) -> th.Tensor:
        if not shared:
            features = self.get_shared(features)
        res = self.value_net(features)
        res = res.view(features.size(0), -1)
        return res


class TransformerActorCriticPolicy(ActorCriticPolicy):
    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            lr_schedule: Callable[[float], float],
            *args,
            **kwargs,
    ):
        # Disable orthogonal initialization
        self.num_offense = action_space.shape[0]
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = TransformerNetwork(self.features_dim, num_offense=self.num_offense)
