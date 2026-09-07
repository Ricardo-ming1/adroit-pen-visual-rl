from __future__ import annotations

import numpy as np
import torch

from src.models.actor import SquashedGaussianActor
from src.state_distillation.oracle_observation import OracleObservationAdapter, rotation_6d_to_matrix
from src.state_distillation.temporal_state_estimator import TemporalStateEstimator


def vision_actor():
    return SquashedGaussianActor(line='vision',cnn_channels=[8,8,8],cnn_kernels=[8,4,3],cnn_strides=[4,2,1],feature_dim=16,hidden_dims=[16,16])


def test_rotation_output_is_valid():
    matrix=rotation_6d_to_matrix(torch.randn(32,6));identity=matrix.transpose(-1,-2)@matrix
    torch.testing.assert_close(identity,torch.eye(3).expand_as(identity),atol=1e-5,rtol=1e-5)
    assert torch.all(torch.linalg.det(matrix)>0.999)


def test_temporal_input_has_no_current_action_slot():
    model=TemporalStateEstimator(vision_actor().visual_encoder)
    assert model.gru.input_size==model.visual_encoder.output_dim+48+24+1
    assert model.sequence_length==8 and not model.uses_privileged_state


def test_adapter_derived_fields_are_closed():
    adapter=OracleObservationAdapter();proprio=torch.randn(3,48);primitive=torch.randn(3,21)
    raw=adapter(proprio,primitive);torch.testing.assert_close(raw[:,:24],proprio[:,:24])
    torch.testing.assert_close(raw[:,39:42],raw[:,24:27]-torch.tensor([0.,-.2,.25]))
    torch.testing.assert_close(raw[:,42:45],raw[:,33:36]-raw[:,36:39])


def test_predicted_path_backpropagates_through_frozen_oracle():
    oracle=SquashedGaussianActor(line='oracle',hidden_dims=[16,16]);oracle.requires_grad_(False)
    primitive=torch.randn(2,21,requires_grad=True);raw=OracleObservationAdapter()(torch.randn(2,48),primitive)
    oracle.act(actor=raw,deterministic=True).sum().backward();assert primitive.grad is not None
    assert all(parameter.grad is None for parameter in oracle.parameters())


def test_episode_padding_does_not_import_another_episode():
    unbounded=np.arange(0-8+1,1);times=np.maximum(unbounded,0);mask=unbounded>=0
    assert np.all(times==0) and mask.sum()==1
