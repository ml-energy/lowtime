import datetime
import logging
import time
import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
plt.rc("svg", hashsalt=None)

from examples.common import df_to_time_costs_pareto, preprocess_time_costs, parse_args
from rene import (
    ReneDAG,
    Synchronous1F1B,
    PDSolver,
    Forward,
    Backward,
    PipelineVisualizer,
    DEFAULT_ANNOTATION_ARGS,
    DEFAULT_LINE_ARGS,
    DEFAULT_RECTANGLE_ARGS,
)


def main():
    # Build an argument parser.
    args = parse_args()
        
    inst_profile = args.inst_profile
    p2p_profile = args.p2p_profile
    output_dir = args.output_dir
    num_mbs = args.num_mbs
    num_stages = args.num_stages
    
    # Instruction offline profiling results.
    inst_df = pd.read_csv(inst_profile)
    time_costs = df_to_time_costs_pareto(inst_df)

    # P2P communication blocking power consumption.
    p2p_block_df = pd.read_csv(p2p_profile)
    p2p_block_df = p2p_block_df.loc[p2p_block_df.time_ms == 100]
    p2p_block_df["power"] = p2p_block_df.energy_mj / p2p_block_df.time_ms / 100

    # Compute the average power consumption of blocking on P2P communication.
    # In the absolute majority of the times we don't go below 800MHz,
    # so we filter frequencies that are below that and take the average so that we're as accurate as possible.
    p_p2p = p2p_block_df.query("freq >= 800").power.mean().item()
    # print(p_p2p)
    time_stamp = datetime.datetime.fromtimestamp(
        time.time()).strftime('%m%d_%H%M%S')
    output_dir = Path(output_dir) / time_stamp
    # output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    log_path = output_dir / "job.log"

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='(%m-%d) %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_path, mode='a'),
            logging.StreamHandler()
        ])
    
    logging.info("Arguments: %s", args)
    # Algorithm part
    interval = args.interval
    unit_time = args.unit_time
    fit_method = args.fit_method
    time_costs = preprocess_time_costs(time_costs, unit_time)
    # print(time_costs)    
    if args.initial_guess:
        with open(args.initial_guess, "r") as f:
            raw_initial_guess = eval(f.read())
            initial_guess = {
                Forward: raw_initial_guess["Forward"],
                Backward: raw_initial_guess["Backward"],
            }
    else:
        initial_guess = {}

    # Instantiate the Instruction DAG.
    dag = ReneDAG(
        schedule_type=Synchronous1F1B,
        num_stages=num_stages,
        num_micro_batches=num_mbs,
        time_costs=time_costs,  # NOTE: This is from inst_df, not sub_p2p_inst_df, because we want to use the original energy to determine colors.
        output_dir=str(output_dir),
        fit_method=fit_method,
        p2p_power=p_p2p,
        initial_guess=initial_guess,
    )

    # Instantiate the visualizer args
    annotation_args = DEFAULT_ANNOTATION_ARGS
    annotation_args[Forward]["fontsize"] = 9.0
    annotation_args[Backward]["fontsize"] = 9.0
    # annotation_args[Backward]["color"] = "#ffffff"
    rectangle_args = DEFAULT_RECTANGLE_ARGS
    rectangle_args[Forward]["hatch"] = "////"
    line_args = DEFAULT_LINE_ARGS
    line_args["linewidth"] = 2.0

    # Instantiate the PD solver.
    pd_solver = PDSolver(dag, output_dir.__str__(), interval, unit_time)
    rene_gen = pd_solver.run()
    prev_cost: float = 0.0
    cost_change: float = 0.0 
    for i, rene_dag in enumerate(rene_gen):
        rene_dag.schedule("eager")
        total_freqs = rene_dag.get_freq_assignment()
        total_cost, refined_cost = rene_dag.get_total_cost()
        cost_change = total_cost - prev_cost
        prev_cost = total_cost
        total_time = rene_dag.get_total_time()
        with open(
            os.path.join(
                output_dir, f"freqs_pipeline_{i:05d}.py"
            ),
            "w",
        ) as f:
            f.write("[\n")
            for freqs in total_freqs:
                f.write(str([int(freq) for freq in freqs]) + ",\n")
            f.write("]\n")
            f.write(
                f"# Iteration {i}: cost change {cost_change} \n"
            )
            f.write(f"# Iteration {i}: total cost {total_cost} \n")
            f.write(
                f"# Iteration {i}: refined cost {refined_cost} \n"
            )
            f.write(
                f"# Iteration {i}: total time {total_time} \n"
            )
        vis = PipelineVisualizer(
        rene_dag,
        annotation_args=annotation_args,
        rectangle_args=rectangle_args,
        line_args=line_args,
        )
        fig, ax = plt.subplots(figsize=(num_mbs * 2, num_stages), tight_layout=dict(pad=0.2, w_pad=0.2, h_pad=0.2))
        vis.draw(ax, draw_time_axis=True, power_color="Oranges")
        vis.draw_critical_path(ax)
        # ax.set_xlim(0, 4.6)  # Fix xlim so that different 1F1B pipelines from different heuristics can be compared side-by-side.
        ax.xaxis.set_label_coords(0.5, -0.07)
        ax.set_xlabel("Time (s)", fontsize=9.0)
        ax.tick_params(axis="x", labelsize=8.0)        
        fig.savefig(os.path.join(output_dir, f"pipeline_{i}.png"), format="PNG")
        plt.clf()
        plt.close()


if __name__ == "__main__":
    main()
