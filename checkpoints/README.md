# Adroit Pen checkpoint bundle

These exact frozen checkpoints are stored with Git LFS. Run `git lfs pull`
after cloning.

| File | Role | SHA-256 |
|---|---|---|
| `vision_awac_seed101.pt` | Frozen visual AWAC actor and training state | `c7034e1f2f18d2853aa8c7f011070a57d4f3e70f3d8fddea4afbc0c6e2a3d07d` |
| `oracle_awac_seed0.pt` | Frozen privileged-state Oracle AWAC | `3b3cee7fdcd4535f9869540d0286c8576a690004ffa24f233cc4a0281f1516fa` |
| `e2_temporal_state_estimator.pt` | Frozen Temporal Visual-State Distillation E2 | `cc35a4130cb33df92fb635e58f125a219da76a2a89fd4109ba7e9267a3080b22` |

E2 is used together with the frozen vision and Oracle actors; it is not a
standalone policy. `configs/v6_1_confirmation.yaml` points to this complete
three-file bundle. Loading uses PyTorch's `weights_only=True` mode.

The E2 policy reads RGB, proprioception, and action history at deployment. Its
privileged state target is training-only. Results are simulation-only and do
not establish real-robot or sim-to-real performance.
