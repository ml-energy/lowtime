import os
import logging
from pathlib import Path

import pandas as pd
from typing import Type
import matplotlib.pyplot as plt
import pickle

from rene import (
    ReneDAG,
    Synchronous1F1B,
    EarlyRecomputation1F1B,
    PDSolver,
    Forward,
    Backward,
    PipelineVisualizer,
    Instruction,
    DEFAULT_ANNOTATION_ARGS,
    DEFAULT_LINE_ARGS,
    DEFAULT_RECTANGLE_ARGS,
)
from examples.common import df_to_time_costs_pareto, preprocess_time_costs, parse_args
from rene.dag import backward_dep, forward_dep, forwardbackward_backward_dep, forwardbackward_dep


def main():
    # Build the argument parser.
    args = parse_args()
        
    # Instruction offline profiling results.
    inst_df = pd.read_csv(args.inst_profile)
    time_costs = df_to_time_costs_pareto(inst_df)

    # P2P communication blocking power consumption.
    if args.p2p_power is None:
        p2p_block_df = pd.read_csv(args.p2p_profile)
        p2p_block_df = p2p_block_df.loc[p2p_block_df.time_ms == 100]
        p2p_block_df["power"] = p2p_block_df.energy_mj / p2p_block_df.time_ms / 100

        # Compute the average power consumption of blocking on P2P communication.
        # In the absolute majority of the times we don't go below 800MHz,
        # so we filter frequencies that are below that and take the average so that we're as accurate as possible.
        p_p2p = p2p_block_df.query("freq >= 800").power.mean().item()
        print(f"Average P2P blocking power consumption: {p_p2p:.2f}W")
    else:
        p_p2p = args.p2p_power
        assert isinstance(p_p2p, float)

    output_dir = Path(args.output_dir)
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

    # Quantize and preprocess the time costs.
    time_costs = preprocess_time_costs(time_costs, args.unit_time)
    # Load the initial guess parameters for instruction exponential curve fitting.
    initial_guess: dict[Type[Instruction], dict[int, list[float]]]
    if args.initial_guess:
        with open(args.initial_guess, "r") as f:
            raw_initial_guess: dict[str, dict[int, list[float]]] = eval(f.read())
            initial_guess = {
                Forward: raw_initial_guess["Forward"],
                Backward: raw_initial_guess["Backward"],
            }
    else:
        initial_guess = {}

    # Instantiate the initial ReneDAG.
    if args.train_scheduel == "1f1b":
        dag = ReneDAG(
            schedule_type=Synchronous1F1B,
            num_stages=args.num_stages,
            num_micro_batches=args.num_mbs,
            time_costs=time_costs,
            p2p_power=p_p2p,
            output_dir=output_dir,
            fit_method=args.fit_method,
            initial_guess=initial_guess,
            unit_time=args.unit_time,
        )
    elif args.train_schedule == "early_recomputation_1f1b":
        dag = ReneDAG(
            schedule_type=EarlyRecomputation1F1B,
            dependency_rules=[forward_dep, backward_dep, forwardbackward_dep, forwardbackward_backward_dep],
            num_stages=args.num_stages,
            num_micro_batches=args.num_mbs,
            time_costs=time_costs,
            p2p_power=p_p2p,
            output_dir=output_dir,
            fit_method=args.fit_method,
            initial_guess=initial_guess,
            unit_time=args.unit_time,
        )

    # Instantiate the visualizer args
    annotation_args = DEFAULT_ANNOTATION_ARGS
    annotation_args[Forward]["fontsize"] = 9.0
    annotation_args[Backward]["fontsize"] = 9.0
    rectangle_args = DEFAULT_RECTANGLE_ARGS
    # rectangle_args[Forward]["hatch"] = "////"
    line_args = DEFAULT_LINE_ARGS
    line_args["linewidth"] = 2.0

    # Instantiate the PD solver.
    pd_solver = PDSolver(dag, output_dir)
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
        with open(output_dir / f"freqs_pipeline_{i:05d}.py", "w") as f:
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
        if i % args.interval == 0:
            # Save the current rene dag using pickle.
            with open(output_dir / f"rene_dag_{i:05d}.pkl", "wb") as f:
                pickle.dump(rene_dag, f)

            vis = PipelineVisualizer(
                rene_dag,
                annotation_args=annotation_args,
                rectangle_args=rectangle_args,
                line_args=line_args,
            )
            fig, ax = plt.subplots(figsize=(args.num_mbs * 2, args.num_stages), tight_layout=dict(pad=0.2, w_pad=0.2, h_pad=0.2))
            vis.draw(ax, draw_time_axis=True, power_color="RdYlGn_r")
            vis.draw_critical_path(ax)
            # ax.set_xlim(0, 4.6)  # Fix xlim so that different 1F1B pipelines from different heuristics can be compared side-by-side.
            ax.xaxis.set_label_coords(0.5, -0.07)
            ax.set_xlabel("Time (s)", fontsize=9.0)
            ax.tick_params(axis="x", labelsize=8.0)        
            fig.savefig(os.path.join(output_dir, f"pipeline_{i}.png"), format="PNG")
            plt.clf()
            plt.close()
        final_rene_dag = rene_dag

    vis = PipelineVisualizer(
        final_rene_dag,
        annotation_args=annotation_args,
        rectangle_args=rectangle_args,
        line_args=line_args,
    )
    fig, ax = plt.subplots(figsize=(args.num_mbs * 2, args.num_stages), tight_layout=dict(pad=0.2, w_pad=0.2, h_pad=0.2))
    vis.draw(ax, draw_time_axis=True, power_color="RdYlGn_r")
    vis.draw_critical_path(ax)
    # ax.set_xlim(0, 4.6)  # Fix xlim so that different 1F1B pipelines from different heuristics can be compared side-by-side.
    ax.xaxis.set_label_coords(0.5, -0.07)
    ax.set_xlabel("Time (s)", fontsize=9.0)
    ax.tick_params(axis="x", labelsize=8.0)        
    fig.savefig(os.path.join(output_dir, f"pipeline_final.png"), format="PNG")
    plt.clf()
    plt.close()        


if __name__ == "__main__":
    main()
