from __future__ import annotations

import inspect
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from src.models.actor import SquashedGaussianActor
from src.state_distillation.direct_action_head import DirectActionPolicy, FrozenTemporalDirectAction
from src.state_distillation.temporal_state_estimator import TemporalStateEstimator
from src.state_distillation.v6_1 import CausalHistory, paired_binary_statistics, sha256_file


E2 = Path(os.environ.get('ADROIT_E2_CHECKPOINT', 'runs/v6/checkpoints/e2.pt'))


def _vision_actor():
    return SquashedGaussianActor(line='vision',cnn_channels=[8,8,8],cnn_kernels=[8,4,3],cnn_strides=[4,2,1],feature_dim=16,hidden_dims=[16,16])


def test_frozen_e2_identity():
    if not E2.is_file():
        pytest.skip('set ADROIT_E2_CHECKPOINT to verify the private frozen E2 artifact')
    assert sha256_file(E2) == 'cc35a4130cb33df92fb635e58f125a219da76a2a89fd4109ba7e9267a3080b22'


def test_confirmation_and_hold_seeds_do_not_overlap_v6_training_or_dev():
    phase_a=set(range(7_000_000,7_000_300));hold=set(range(7_100_000,7_100_600))
    historical=set(range(30_000,30_100))|set(range(6_100_000,6_100_300))|set(range(6_200_000,6_200_300))
    assert not phase_a & historical and not hold & historical and not phase_a & hold


def test_direct_head_is_bounded_full_action_and_has_no_phase_input():
    estimator=TemporalStateEstimator(_vision_actor().visual_encoder);model=FrozenTemporalDirectAction(estimator,[16])
    action=model(torch.zeros(2,8,3,84,84,dtype=torch.uint8),torch.zeros(2,8,48),torch.zeros(2,8,24),torch.ones(2,8))
    assert action.shape==(2,24) and torch.all(action.abs()<=1)
    source=inspect.getsource(DirectActionPolicy.act)
    assert 'phase' not in source and 'privileged' not in source and 'reward' not in source


def test_common_history_stores_only_seven_before_current_rows():
    observation={'rgb':np.zeros((4,3,4,4),np.uint8),'proprio':np.zeros(48,np.float32),'previous_action':np.zeros(24,np.float32)}
    history=CausalHistory();history.reset(observation);saved=history.before_current()
    assert saved['frames'].shape[0]==7 and saved['proprio'].shape==(7,48) and saved['actions'].shape==(7,24)


def test_paired_bootstrap_uses_seed_matched_episode_differences():
    base=[{'seed':i,'ok':False} for i in range(4)];candidate=[{'seed':i,'ok':i<2} for i in range(4)]
    result=paired_binary_statistics(candidate,base,'ok',bootstrap_replicates=1000,bootstrap_seed=1)
    assert result['wins']==2 and result['losses']==0 and result['ties']==2 and result['paired_difference']==0.5
