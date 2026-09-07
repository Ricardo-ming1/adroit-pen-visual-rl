from __future__ import annotations

from collections import deque
import inspect

import numpy as np

from scripts import collect_state_dagger
from src.state_distillation.evaluation import DistilledOraclePolicy


def test_gru_history_reset_at_episode_boundary():
    policy = DistilledOraclePolicy.__new__(DistilledOraclePolicy)
    policy.frames = deque([np.ones((3, 2, 2))], maxlen=8)
    policy.proprio = deque([np.ones(48)], maxlen=8)
    policy.actions = deque([np.ones(24)], maxlen=8)
    policy.valid = deque([1.0], maxlen=8)
    policy.reset()
    assert not policy.frames and not policy.proprio and not policy.actions and not policy.valid


def test_dagger_executes_student_and_only_records_teacher_label():
    source = inspect.getsource(collect_state_dagger.collect)
    assert "action,_=policy.act(observation)" in source
    assert "teacher=deterministic_action(oracle" in source
    assert "env.step(action)" in source
    assert "env.step(teacher)" not in source


def test_deployment_policy_has_no_privileged_interface():
    source = inspect.getsource(DistilledOraclePolicy.act)
    assert "privileged" not in source
    assert "reward" not in source and "success" not in source
