from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from gymnasium import spaces
import torch as th
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.policies import ActorCriticPolicy
import torch as th
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)
from torch import nn

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
            last_layer_dim_pi: int = 64,
            last_layer_dim_vf: int = 64,
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
            # nn.Linear(d_model, last_layer_dim_pi)
        )

        # Value network
        self.value_net = nn.Sequential(
            *[nn.TransformerEncoderLayer(d_model=d_model, nhead=1, dim_feedforward=64, dropout=0.1, layer_norm_eps=1e-05, batch_first=True, norm_first=False,
                                         bias=True) for _ in range(1)],
            # nn.Linear(d_model, last_layer_dim_pi)
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
        res = res.mean(dim=-2)
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
        self.last_layer_dim_pi = 64
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        self.action_net = nn.Linear(self.last_layer_dim_pi, self.action_space[0].n)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = TransformerNetwork(self.features_dim, num_offense=self.num_offense, last_layer_dim_pi=self.last_layer_dim_pi)

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        b = latent_pi.size(0)
        latent_pi = latent_pi.view(b * self.num_offense, -1)
        mean_actions = self.action_net(latent_pi).view(b, -1)

        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_pi)
        else:
            raise ValueError("Invalid action distribution")
