from typing import Type
import pandas as pd
import matplotlib.pyplot as plt

from rene import (
    Instruction,
    Forward,
    Backward,
)

TIME_COST_T = dict[Type[Instruction], dict[int, list[tuple[float, float, int]]]]

def df_to_time_cost_pareto(inst_df: pd.DataFrame) -> TIME_COST_T:
    """Filter a raw Instruction dataframe profile into a new dataframe containing only entries on the Pareto frontier"""
    time_costs: TIME_COST_T = {Forward: {}, Backward: {}}

    # Leave only Pareto-optimal points.
    for stage in range(inst_df.stage.max() + 1):
        for instruction in ["forward", "backward"]:
            _df = inst_df[(inst_df.stage == stage) & (inst_df.instruction == instruction)]
            fs = _df.frequency.to_list()
            ts = _df.time.to_list()
            es = _df.energy.to_list()
            opt_fs, opt_ts, opt_es = [], [], []
            for f1, t1, e1 in zip(fs, ts, es):
                if any(t_new < t1 and e_new < e1 for t_new, e_new in zip(ts, es)):
                    continue
                opt_fs.append(f1)
                opt_ts.append(t1)
                opt_es.append(e1)

            time_costs[Forward if instruction == "forward" else Backward][stage] = list(
                zip(opt_ts, opt_es, opt_fs)
            )

    return time_costs

# Subtract P2P energy: E - P_P2P * T
def subtract_p2p(inst_df, p2p_df):
    """Subtract energy profile in inst_df with p2p energy in p2p_df"""
    _df = p2p_df.loc[p2p_df.time_ms >= 100]
    p2p_power = {
        freq: _df.loc[_df.freq == freq].power.to_list()
        for freq in _df.freq.unique()
    }
    p2p_power = {freq: sum(power) / len(power) for freq, power in p2p_power.items()}
    def subtract_p2p(row):
        row.energy -= row.time * p2p_power[row.frequency]
        return row
    return inst_df.apply(subtract_p2p, axis=1)