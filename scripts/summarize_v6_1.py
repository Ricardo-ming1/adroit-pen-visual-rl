from __future__ import annotations

import json
from pathlib import Path

from src.utils import load_yaml, save_json


def maybe(path):
    path=Path(path);return json.loads(path.read_text()) if path.exists() else None


def percent(value):
    return f"{100*float(value):.1f}%"


def main():
    c=load_yaml('configs/v6_1_confirmation.yaml');root=Path(c['results_dir'])
    phase=maybe(root/'phase_a_confirmation300.json');hold=maybe(root/'fixed_entry_hold200.json');route=maybe(root/'route_decision.json')
    reproduction=maybe(root/'reproduction'/'three_seed_reproduction.json');repro_hold=maybe(root/'reproduction'/'three_seed_fixed_hold.json')
    frozen=maybe(root/'frozen_test.json');direct_train=maybe(root/'direct_action_training.json');direct_dev=maybe(root/'direct_action_development100.json');direct_confirm=maybe(root/'direct_action_confirmation300.json')
    if phase is None or hold is None or route is None:raise RuntimeError('V6.1 required Phase A results are incomplete')
    if not phase['phase_a_confirmation_pass']:status='STOP_V6_1_E2_INDEPENDENT_CONFIRMATION_NOT_MET'
    elif reproduction is None:status='V6_1_E2_CONFIRMED_REPRODUCTION_INCOMPLETE'
    elif not reproduction['three_seed_gate_pass']:status='STOP_V6_1_THREE_SEED_REPRODUCTION_GATE_NOT_MET'
    elif frozen is None:status='V6_1_THREE_SEED_PASS_FROZEN_TEST_NOT_RUN'
    else:status='V6_1_E2_CONFIRMATION_REPRODUCTION_AND_FROZEN_TEST_COMPLETE'
    summary={'final_status':status,'v6_formal_status':'STOP_V6_PROMOTION_GATE_NOT_MET','v6_history_modified':False,
        'phase_a':phase,'fixed_entry_hold':hold,'route_decision':route,'direct_action_rescue_executed':direct_train is not None,
        'direct_action_development':direct_dev,'direct_action_confirmation':direct_confirm,'three_seed_reproduction':reproduction,
        'three_seed_fixed_hold':repro_hold,'frozen_test':frozen,'final_deployable_candidate_checkpoint':c['e2_checkpoint'],
        'final_deployable_candidate_checkpoint_sha256':c['e2_checkpoint_sha256']}
    save_json(root/'summary.json',summary)
    p=phase['policies'];b=phase['paired_e2_vs_e0']['benchmark'];s=phase['paired_e2_vs_e0']['strict'];h=hold['policies']
    lines=['# V6.1 E2 Independent Confirmation and Hold Rescue','',f'Final status: `{status}`','',
        'V6 remains unchanged: `STOP_V6_PROMOTION_GATE_NOT_MET`. V6.1 fixed the E2 checkpoint before opening any new confirmation data.','',
        '## Independent 300-episode confirmation','',
        '| Policy | Benchmark | Strict | Entry | Exit / entry | 20-step hold |','|---|---:|---:|---:|---:|---:|']
    for name,label in [('e0','Frozen Vision AWAC'),('e2','Frozen E2'),('oracle','True-state Oracle')]:
        row=p[name];lines.append(f"| {label} | {row['benchmark_success_count']}/300 ({percent(row['benchmark_success'])}) | {row['strict_success_count']}/300 ({percent(row['strict_stable_success'])}) | {row['goal_entry_count']}/300 | {row['goal_exit_count']}/{row['goal_entry_count']} ({percent(row['goal_exit_after_entry_rate'])}) | {row['twenty_step_hold_count']}/300 ({percent(row['twenty_step_hold_rate'])}) |")
    lines += ['',f"E2–E0 Benchmark: {percent(phase['benchmark_difference'])}; paired wins/losses/ties {b['wins']}/{b['losses']}/{b['ties']}, 95% bootstrap CI {percent(b['paired_bootstrap_95_ci'][0])} to {percent(b['paired_bootstrap_95_ci'][1])}, exact McNemar p={b['exact_mcnemar_two_sided_p']:.3g}.",
        f"E2–E0 Strict: {percent(phase['strict_difference'])}; paired wins/losses/ties {s['wins']}/{s['losses']}/{s['ties']}, CI {percent(s['paired_bootstrap_95_ci'][0])} to {percent(s['paired_bootstrap_95_ci'][1])}, p={s['exact_mcnemar_two_sided_p']:.3g}.",'',
        '## Fixed E0-entry hold roots','', '| Policy | 20-step survival | 50-step survival | Mean steps to exit |','|---|---:|---:|---:|']
    for name,label in [('e0','Frozen Vision'),('e2','Frozen E2'),('oracle','True-state Oracle')]:
        row=h[name];lines.append(f"| {label} | {row['twenty_step_survival_count']}/200 | {row['fifty_step_survival_count']}/200 | {row['mean_steps_to_exit']:.2f} |")
    lines += ['',f"Route: `{route['selected_route']}`. Direct-action rescue executed: `{direct_train is not None}`."]
    if reproduction:
        lines += ['','## Three-seed reproduction','', '| Recipe seed | Benchmark | Strict | Entry | 20-step hold |','|---:|---:|---:|---:|---:|']
        for row in reproduction['seeds']:lines.append(f"| {row['recipe_seed']} | {percent(row['benchmark_success'])} | {percent(row['strict_stable_success'])} | {percent(row['goal_entry_rate'])} | {percent(row['twenty_step_hold_rate'])} |")
        lines += ['',f"Mean Benchmark {percent(reproduction['benchmark_mean'])} ± {percent(reproduction['benchmark_std'])}; mean Strict {percent(reproduction['strict_mean'])} ± {percent(reproduction['strict_std'])}. Gate pass: `{reproduction['three_seed_gate_pass']}`."]
    if frozen:lines += ['','## Frozen test','',f"Frozen E2 Benchmark {percent(frozen['e2']['benchmark_success'])}; Strict {percent(frozen['e2']['strict_stable_success'])}. The bank was not used for selection or retraining."]
    lines += ['','## Reproduction','', '```bash','export PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl',
        'python scripts/confirm_e2.py','python scripts/build_fixed_entry_hold_bank.py','python scripts/evaluate_fixed_entry_hold.py',
        'python scripts/reproduce_e2_seed.py --recipe-seed 1606','python scripts/reproduce_e2_seed.py --recipe-seed 2606','```','',
        'Large checkpoints, root states, and partial episode outputs remain under ignored `runs/v6_1/`.']
    (root/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'final_status':status,'summary':str(root/'summary.json'),'readme':str(root/'README.md')},indent=2))


if __name__=='__main__':main()
