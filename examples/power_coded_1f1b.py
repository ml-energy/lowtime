import shutil
import argparse
from typing import Type
from matplotlib.patches import Patch
from matplotlib.colors import Normalize

import pandas as pd
import matplotlib.pyplot as plt
plt.rc("svg", hashsalt=None)

from rene import (
    ReneDAGOld,
    Synchronous1F1B,
    Instruction,
    Forward,
    Backward,
    PipelineVisualizer,
    DEFAULT_ANNOTATION_ARGS,
    DEFAULT_RECTANGLE_ARGS,
    DEFAULT_LINE_ARGS,
)


TIME_COST_T = dict[Type[Instruction], dict[int, list[tuple[float, float, int]]]]


parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="h_hybrid")
args = parser.parse_args()


def df_to_time_costs_pareto(inst_df: pd.DataFrame) -> TIME_COST_T:
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
inst_df = pd.read_csv("../perseus-analysis/data/merak_offline_profiler/A40/dp1+pp4+tp1/merak+gpt3-large+uniform_transformer+dp1+pp4+tp1+mbs4.csv")
time_costs = df_to_time_costs_pareto(inst_df)

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
    # p2p_power = {freq: sum(power) / len(power) for freq, power in p2p_power.items()}
    p2p_power = 75.5
    def subtract_p2p(row):
        row.energy -= row.time * p2p_power  #[row.frequency]
        return row
    return inst_df.apply(subtract_p2p, axis=1)
sub_p2p_inst_df = subtract_p2p(inst_df, p2p_block_df)

# Instantiate the Instruction DAG.
dag = ReneDAGOld(
    schedule_type=Synchronous1F1B,
    num_stages=4,
    num_micro_batches=8,
    time_costs=time_costs,  # NOTE: This is from inst_df, not sub_p2p_inst_df, because we want to use the original energy to determine colors.
    unit_time=1.0,
)

# Set instruction frequencies
name = args.name
cmap = "RdBu_r"
bottleneck_stage = 3
norm = Normalize(vmin=-150, vmax=500)

# 1) Max frequency for all instructions
if name == "maxfreq":
    for inst in dag.insts:
        _df = inst_df.query(f"stage == {inst.stage_id} and instruction == '{type(inst).__name__.lower()}' and frequency == 1740").iloc[0]
        inst.duration, inst.cost, inst.frequency = _df.time, _df.energy, _df.frequency
# 2) H_balanced frequency for all instructions
# For instructions on the critical path, choose the maximum frequency.
# For others, choose the the frequency that balances computation time with the bottleneck stage.
elif name == "h_balanced":
    target_forward_time = min(time_costs[Forward][bottleneck_stage])[0]
    target_backward_time = min(time_costs[Backward][bottleneck_stage])[0]
    for inst in dag.insts:
        # Critical path ops (last stage bottleneck assumed)
        if inst.stage_id == bottleneck_stage or (inst.micro_batch_id == 0 and isinstance(inst, Forward)) or (inst.micro_batch_id == dag.num_micro_batches - 1 and isinstance(inst, Backward)):
            _df = inst_df.query(f"stage == {inst.stage_id} and instruction == '{type(inst).__name__.lower()}' and frequency == 1740").iloc[0]
            inst.duration, inst.cost, inst.frequency = _df.time, _df.energy, _df.frequency
            continue
        # Non-critical path ops: Choose the frequency that balances computation time with the bottleneck stage.
        target_time = target_forward_time if isinstance(inst, Forward) else target_backward_time
        _df = inst_df.query(f"stage == {inst.stage_id} and instruction == '{type(inst).__name__.lower()}' and time <= {target_time}")
        balanced_time_cost = _df.iloc[_df.frequency.argmin()]  # Smaller frequency is always longer time.
        inst.duration, inst.cost, inst.frequency = balanced_time_cost.time, balanced_time_cost.energy, balanced_time_cost.frequency
# 3) H_minenergy frequency for all instructions
# For instructions on the critical path, choose the maximum frequency.
# For others, choose the frequency that consumes the minimum sub_p2p_energy.
elif name == "h_minenergy":
    for inst in dag.insts:
        # Critical path ops (last stage bottleneck assumed)
        if inst.stage_id == bottleneck_stage or (inst.micro_batch_id == 0 and isinstance(inst, Forward)) or (inst.micro_batch_id == dag.num_micro_batches - 1 and isinstance(inst, Backward)):
            _df = inst_df.query(f"stage == {inst.stage_id} and instruction == '{type(inst).__name__.lower()}' and frequency == 1740").iloc[0]
            inst.duration, inst.cost, inst.frequency = _df.time, _df.energy, _df.frequency
            continue
        # Other ops: Frequency that minimizes energy from sub_p2p_inst_df.
        _df = sub_p2p_inst_df.query(f"stage == {inst.stage_id} and instruction == '{type(inst).__name__.lower()}'")
        min_sub_p2p_cost_freq = _df.iloc[_df.energy.argmin()].frequency
        # Choose the frequency closest to min_sub_p2p_cost_freq from inst.time_costs.
        min_sub_p2p_time_cost = inst_df.query(f"stage == {inst.stage_id} and instruction == '{type(inst).__name__.lower()}' and frequency == {min_sub_p2p_cost_freq}").iloc[0]
        inst.duration, inst.cost, inst.frequency = min_sub_p2p_time_cost.time, min_sub_p2p_time_cost.energy, min_sub_p2p_time_cost.frequency
# 4) H_bybrid frequency for all instructions
# For instructions on the critical path, choose the maximum frequency.
# For others, choose the *larger* between f_bal (the frequency that balances computation time with the bottleneck stage) and f_min (the frequency that consumes the minimum sub_p2p_energy).
# - case 1: f_bal <= f_min => All good. We choose f_min, which consumes less energy. This will leave some idle time, but that's fine.
# - case 2: f_bal > f_min => Ideally we want to choose f_min because it consumes less energy, but it may lengthen the critical path. Choose f_bal.
elif name == "h_hybrid":
    target_forward_time = min(time_costs[Forward][bottleneck_stage])[0]
    target_backward_time = min(time_costs[Backward][bottleneck_stage])[0]
    for inst in dag.insts:
        # Critical path ops (last stage bottleneck assumed)
        if inst.stage_id == bottleneck_stage or (inst.micro_batch_id == 0 and isinstance(inst, Forward)) or (inst.micro_batch_id == dag.num_micro_batches - 1 and isinstance(inst, Backward)):
            _df = inst_df.query(f"stage == {inst.stage_id} and instruction == '{type(inst).__name__.lower()}' and frequency == 1740").iloc[0]
            inst.duration, inst.cost, inst.frequency = _df.time, _df.energy, _df.frequency
            continue
        # Other ops
        # a) Frequency that minimizes energy from sub_p2p_time_costs
        _df = sub_p2p_inst_df.query(f"stage == {inst.stage_id} and instruction == '{type(inst).__name__.lower()}'")
        min_sub_p2p_cost_freq = _df.iloc[_df.energy.argmin()].frequency
        # Choose the frequency closest to min_sub_p2p_cost_freq from inst_df.
        _df = inst_df.query(f"stage == {inst.stage_id} and instruction == '{type(inst).__name__.lower()}' and frequency == {min_sub_p2p_cost_freq}")
        freq_dist = _df.frequency - min_sub_p2p_cost_freq
        assert freq_dist.min() >= 0
        min_sub_p2p_time_cost = _df.iloc[freq_dist.argmin()]
        min_sub_p2p_time_cost = min_sub_p2p_time_cost.time, min_sub_p2p_time_cost.energy, min_sub_p2p_time_cost.frequency
        # b) Frequency that balances computation time with the bottleneck stage
        target_time = target_forward_time if isinstance(inst, Forward) else target_backward_time
        _df = inst_df.query(f"stage == {inst.stage_id} and instruction == '{type(inst).__name__.lower()}' and time <= {target_time}")
        balanced_time_cost = _df.iloc[_df.frequency.argmin()]
        balanced_time_cost = balanced_time_cost.time, balanced_time_cost.energy, balanced_time_cost.frequency
        # Choose the larger frequency of a) and b)
        inst.duration, inst.cost, inst.frequency = max(min_sub_p2p_time_cost, balanced_time_cost, key=lambda x: x[2])

        if inst.frequency != balanced_time_cost[2]:
            print(f"{inst} frequency: {inst.frequency} (min_sub_p2p: {min_sub_p2p_time_cost[2]}, balanced: {balanced_time_cost[2]})")

dag.annotate_nodes()

# Schedule instructions with the "eager" scheduling algorithm.
# Refer to the docstring for available scheduling algorithms.
dag.schedule("eager")

# Pass the DAG to the pipeline visualizer.
# Refer to the constructor for matplotlib customization.
annotation_args = DEFAULT_ANNOTATION_ARGS
annotation_args[Forward]["fontsize"] = 9.0
annotation_args[Backward]["fontsize"] = 9.0
annotation_args[Forward]["color"] = "black"
annotation_args[Backward]["color"] = "black"
rectangle_args = DEFAULT_RECTANGLE_ARGS
# rectangle_args[Forward]["hatch"] = "////"
line_args = DEFAULT_LINE_ARGS
line_args["linewidth"] = 2.0
vis = PipelineVisualizer(
    dag,
    annotation_args=annotation_args,
    rectangle_args=rectangle_args,
    line_args=line_args,
)

def annotation_hook(inst: Instruction) -> str:
    return f"{'F' if isinstance(inst, Forward) else 'B'}\n{inst.micro_batch_id + 1}"

# Instantitate a matplotlib subplot and draw the pipeline and critical path.
# if args.name == "maxfreq":
#     figsize = (4.5, 1.95)
# else:
figsize = (4.5, 2.1)
fig, ax = plt.subplots(figsize=figsize, tight_layout=dict(pad=0.2, w_pad=0.2, h_pad=0.2))
vis.draw(ax, draw_time_axis=True, annotation_hook=annotation_hook, power_color=cmap, normalizer=norm)
# vis.draw_critical_path(ax)
# ax.set_xlim(0, 4.6)  # Fix xlim so that different 1F1B pipelines from different heuristics can be compared side-by-side.
ax.xaxis.set_label_coords(0.5, -0.07)
ax.set_xlabel("Time (s)", fontsize=9.0)
ax.tick_params(axis="x", labelsize=8.0)

# Legend: Hatched boxes are forward, solid boxes are backward.
# if args.name == "maxfreq":
#     legend_elements = [
#         # plt.Line2D([0], [0], color="black", lw=2, label="Critical path"),
#         Patch(facecolor="white", edgecolor="black", hatch="////", label="Forward"),
#         Patch(facecolor="white", edgecolor="black", label="Backward"),
#     ]
#     ax.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=2, fontsize=8.0, frameon=False)

# Below the legend, draw a colorbar for power consumption.
# if args.name != "maxfreq":
# norm = Normalize(vmin=0, vmax=max_power)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", location="top", aspect=50, ticks=[0, 75.5, 150, 300], fraction=0.044)
# cbar.ax.set_title("Power (W)", fontsize=9.0)
cbar.ax.set_xlim(0, 300)
cbar.ax.set_xticklabels(["0W", "75.5W ($P_{\mathrm{P2P}}$)", "150W", "300W"], fontsize=9.0)

# Add text annotations for the stages along the y axis.
for stage in range(4):
    ax.text(-0.1, stage + 0.5, f"S{stage+1}", fontsize=9.0, ha="right", va="center")

prefix = f"figures/power_pipeline_draft+{name}+merak+gpt3-large+uniform_transformer+dp1+tp1+pp4+mbs4"
fig.savefig(f"{prefix}.png")
fig.savefig(f"{prefix}.svg", metadata={"Date": None})
fig.savefig(f"{prefix}.pdf", metadata={"CreationDate": None})
shutil.copyfile(f"{prefix}.pdf", f"../perseus-paper/{prefix}.pdf")


total_time = dag.get_total_time()
stage_times = [0.0 for _ in range(dag.num_stages)]
stage_inst_costs = [0.0 for _ in range(dag.num_stages)]
p2p_power = 75.5
for inst in dag.insts:
    print(f"{inst}: {inst.frequency} MHz")
    stage_times[inst.stage_id] += inst.duration
    stage_inst_costs[inst.stage_id] += inst.cost
total_cost = 0.0
for stage_id in range(dag.num_stages):
    total_cost += stage_inst_costs[stage_id]
    total_cost += (total_time - stage_times[stage_id]) * p2p_power
print(name)
print("Total execution time:", total_time)
print("Total execution cost:", total_cost)
