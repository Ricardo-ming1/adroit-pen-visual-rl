from pathlib import Path

import pytest

from src.data.offline import PenOfflineData


DATASET = Path("data/processed/pen_human_visual.hdf5")


@pytest.mark.skipif(not DATASET.exists(), reason="run scripts.build_visual_dataset first")
def test_no_cross_episode_transition():
    data = PenOfflineData(DATASET)
    transitions = data.indices("all", transitions=True)
    assert len(transitions) == 4975
    assert (transitions[:, 1] < 199).all()
    assert data.valid_transitions == 4975
    for episode_axis in range(25):
        times = transitions[transitions[:, 0] == episode_axis, 1]
        assert times.tolist() == list(range(199))
