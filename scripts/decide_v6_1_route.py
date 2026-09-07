from __future__ import annotations

import json
from pathlib import Path

from src.utils import load_yaml, save_json


def main():
    config=load_yaml('configs/v6_1_confirmation.yaml')
    phase=json.loads((Path(config['results_dir'])/'phase_a_confirmation300.json').read_text())
    hold=json.loads((Path(config['results_dir'])/'fixed_entry_hold200.json').read_text())
    trigger=config['phase_b_trigger']
    acquisition_pattern=(phase['benchmark_difference']>=float(trigger['minimum_benchmark_lift']) and
        phase['strict_difference']<=float(trigger['maximum_strict_lift_for_acquisition_pattern']))
    hold_gap=bool(hold['oracle_hold_gap_triggered'])
    rescue=bool(phase['phase_a_confirmation_pass'] and acquisition_pattern and hold_gap)
    if not phase['phase_a_confirmation_pass']:
        route='STOP_PHASE_A_NOT_CONFIRMED'
    elif rescue:
        route='PHASE_B_PHASE_BALANCED_DIRECT_ACTION_RESCUE'
    else:
        route='SKIP_RESCUE_REPRODUCE_FROZEN_E2_THREE_SEEDS'
    result={'phase_a_confirmation_pass':phase['phase_a_confirmation_pass'],'benchmark_difference':phase['benchmark_difference'],
        'strict_difference':phase['strict_difference'],'fixed_root_oracle_hold_gap_triggered':hold_gap,
        'acquisition_without_strict_improvement_pattern':acquisition_pattern,'phase_b_rescue_authorized':rescue,
        'selected_route':route,'decision_frozen_before_reproduction':True}
    save_json(Path(config['results_dir'])/'route_decision.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
