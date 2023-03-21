import shutil
import argparse
from typing import Type
from matplotlib.patches import Patch
from matplotlib.colors import Normalize

import pandas as pd
import matplotlib.pyplot as plt
plt.rc("svg", hashsalt=None)

from rene import (
    CriticalDAG,
    Synchronous1F1B,
    Instruction,
    Forward,
    Backward,
    PipelineVisualizer,
    PD_Solver,
    DEFAULT_ANNOTATION_ARGS,
    DEFAULT_RECTANGLE_ARGS,
    DEFAULT_LINE_ARGS,
)


TIME_COST_T = dict[Type[Instruction], dict[int, list[tuple[float, float, int]]]]


parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="h_hybrid")
args = parser.parse_args()


def df_to_time_cost_pareto(inst_df: pd.DataFrame) -> TIME_COST_T:
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


# Instruction offline profiling results.
inst_df = pd.read_csv("/Users/yilegu/Desktop/research/SymbioticLab/Perseus/perseus-analysis/data/merak_offline_profiler/merak+gpt3-large+uniform_transformer+dp1+tp1+pp4+mbs4.csv")
time_costs = df_to_time_cost_pareto(inst_df)

# P2P communication blocking power consumption.
p2p_block_df = pd.read_csv("../perseus-analysis/data/p2p-benchmark/intranode-bare-nvlink-sleep-1665-0-1.csv")
p2p_block_df = p2p_block_df.loc[p2p_block_df.time_ms == 100]
p2p_block_df["power"] = p2p_block_df.energy_mj / p2p_block_df.time_ms / 100
# Subtract P2P energy: E - P_P2P * T
def subtract_p2p(inst_df, p2p_df):
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
sub_p2p_inst_df = subtract_p2p(inst_df, p2p_block_df)

# Instantiate the Instruction DAG.
dag = CriticalDAG(
    schedule_type=Synchronous1F1B,
    num_stages=4,
    num_micro_batches=128,
    time_costs=time_costs,  # NOTE: This is from inst_df, not sub_p2p_inst_df, because we want to use the original energy to determine colors.
)

pd_solver = PD_Solver(dag, "/Users/yilegu/Desktop/research/SymbioticLab/Perseus/rene/results/gpt3_pp4_dp128")
pd_solver.run_pd_algorithm()

