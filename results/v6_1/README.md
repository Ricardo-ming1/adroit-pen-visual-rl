# V6.1 E2 Independent Confirmation and Hold Rescue

Final status: `V6_1_E2_CONFIRMATION_REPRODUCTION_AND_FROZEN_TEST_COMPLETE`

V6 remains unchanged: `STOP_V6_PROMOTION_GATE_NOT_MET`. V6.1 fixed the E2 checkpoint before opening any new confirmation data.

## Independent 300-episode confirmation

| Policy | Benchmark | Strict | Entry | Exit / entry | 20-step hold |
|---|---:|---:|---:|---:|---:|
| Frozen Vision AWAC | 157/300 (52.3%) | 81/300 (27.0%) | 157/300 | 93/157 (59.2%) | 118/300 (39.3%) |
| Frozen E2 | 205/300 (68.3%) | 111/300 (37.0%) | 205/300 | 120/205 (58.5%) | 170/300 (56.7%) |
| True-state Oracle | 194/300 (64.7%) | 109/300 (36.3%) | 194/300 | 99/194 (51.0%) | 169/300 (56.3%) |

E2–E0 Benchmark: 16.0%; paired wins/losses/ties 66/18/216, 95% bootstrap CI 10.3% to 21.7%, exact McNemar p=1.33e-07.
E2–E0 Strict: 10.0%; paired wins/losses/ties 60/30/210, CI 4.0% to 16.0%, p=0.00206.

## Fixed E0-entry hold roots

| Policy | 20-step survival | 50-step survival | Mean steps to exit |
|---|---:|---:|---:|
| Frozen Vision | 123/200 | 100/200 | 32.86 |
| Frozen E2 | 123/200 | 94/200 | 32.81 |
| True-state Oracle | 129/200 | 106/200 | 34.27 |

Route: `SKIP_RESCUE_REPRODUCE_FROZEN_E2_THREE_SEEDS`. Direct-action rescue executed: `False`.

## Three-seed reproduction

| Recipe seed | Benchmark | Strict | Entry | 20-step hold |
|---:|---:|---:|---:|---:|
| 606 | 65.0% | 40.0% | 65.0% | 59.0% |
| 1606 | 61.0% | 37.0% | 61.0% | 52.0% |
| 2606 | 57.0% | 28.0% | 57.0% | 47.0% |

Mean Benchmark 61.0% ± 4.0%; mean Strict 35.0% ± 6.2%. Gate pass: `True`.

## Frozen test

Frozen E2 Benchmark 75.0%; Strict 33.5%. The bank was not used for selection or retraining.

## Reproduction

```bash
export PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl
python scripts/confirm_e2.py
python scripts/build_fixed_entry_hold_bank.py
python scripts/evaluate_fixed_entry_hold.py
python scripts/reproduce_e2_seed.py --recipe-seed 1606
python scripts/reproduce_e2_seed.py --recipe-seed 2606
```

Large checkpoints, root states, and partial episode outputs remain under ignored `runs/v6_1/`.
