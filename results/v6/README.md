# V6 Temporal Visual State Distillation

V6 reconstructs the frozen Oracle AWAC observation from eight causal RGB frames,
deployable hand proprioception, and prior executed actions, then runs that one
estimated-state Oracle controller continuously from the first episode step.

## Observation contract

The environment's 45D Oracle observation is not an opaque "45D state". It is:

1. hand `qpos[:24]`, copied from deployable proprioception;
2. world-frame object position and world-frame 6D twist, predicted;
3. object and target SO(3), represented internally as column-major rotation 6D;
4. object/target local `+Z` axes, derived from the projected rotation matrices;
5. object minus fixed target position `(0, -0.2, 0.25)` and the axis difference,
   deterministically reconstructed.

True simulator state passed through this adapter matches the native observation to
`1.2e-7` and the frozen Oracle action to `6.8e-7`. The same validation bank exactly
reproduced 75% Benchmark / 42% Strict.

## Results

See [summary.csv](summary.csv). E2 reached 78% Benchmark and 38% Strict, but its
62.82% goal-exit-after-entry rate exceeded the allowed baseline-plus-3pp limit by
0.143 percentage points. D1 restored exit behavior but lost task performance.
Consequently V6 did not establish a promotion-qualified closed-loop improvement.

V5's one-time privileged closure pilot produced four gains and one harm over 200
paired development episodes (net utility `4 - 2*1 = 2`), with no Benchmark change
and +1.5pp Strict. It is diagnostic only and the candidate-routing line is closed.

## Reproduction

```bash
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl
python scripts/validate_oracle_observation_adapter.py
python scripts/evaluate_state_distilled_policy.py --output results/v6/e0_frozen_vision_validation100.json
python scripts/build_state_distillation_dataset.py --output runs/v6/source_sequences.h5
python scripts/train_state_estimator.py --stage e1 --data runs/v6/source_sequences.h5
python scripts/evaluate_state_distilled_policy.py --checkpoint runs/v6/checkpoints/e1.pt --output results/v6/e1_validation100.json --baseline-output results/v6/e0_frozen_vision_validation100.json
python scripts/train_state_estimator.py --stage e2 --data runs/v6/source_sequences.h5 --initialize runs/v6/checkpoints/e1.pt
python scripts/collect_state_dagger.py --round d1 --checkpoint runs/v6/checkpoints/e2.pt --output runs/v6/d1_sequences.h5
python scripts/train_state_estimator_balanced.py --round d1 --source runs/v6/source_sequences.h5 --recent runs/v6/d1_sequences.h5 --initialize runs/v6/checkpoints/e2.pt
```

Large datasets and checkpoints remain under ignored `runs/v6/`. No V1-V5 artifact,
Frozen Vision actor, Frozen Oracle actor, or Oracle normalizer is modified.
