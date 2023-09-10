from __future__ import annotations

import logging
import argparse
from pathlib import Path
from pprint import pprint
from matplotlib import pyplot as plt

import pandas as pd

from rene.operation import CandidateExecutionOptions, OperationSpec, ExecutionOption, ExponentialModel


logger = logging.getLogger()

def main() -> None:
    args = parse_args()

    # Instruction offline profiling results.
    inst_df = pd.read_csv(args.inst_profile)

    # If specified, merge in online-profiled execution time data.
    if args.replace_time is not None:
        new_time_df = pd.read_csv(args.replace_time)
        # get smallest freq in new_time_df, this will be the cutoff for the inst_df
        min_freq = new_time_df.frequency.min()
        inst_df = inst_df[inst_df.frequency >= min_freq]
        inst_df = inst_df.merge(new_time_df, on=["stage", "instruction", "frequency"], how="left", suffixes=('_x', '_y'))
        inst_df["time_x"] = inst_df["time_y"]
        inst_df = inst_df.drop(columns=["time_y"])
        inst_df = inst_df.rename(columns={"time_x": "time"})

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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    log_path = output_dir / "job.log"

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_path, mode='a'),
            logging.StreamHandler()
        ])
    
    logger.info("Arguments: %s", args)

    operation_dict = {}
    for instruction in ["forward", "backward"]:
        for stage in range(args.num_stages):
            logger.info(f"Processing {instruction} stage {stage}")
            options = []
            _df = inst_df.query("stage == @stage and instruction == @instruction")
            for _, row in _df.iterrows():
                options.append(
                    ExecutionOption[int](
                        real_time=row["time"],
                        unit_time=0.001,
                        cost=row["energy"],
                        knob=row["frequency"]
                    )
                )
            
            # Get the Preto frontier, quantize time, and deduplicate time.
            cand_options = CandidateExecutionOptions[int](options=options)

            # Remap the cost to be effective computation energy.
            # Everything from now on is in terms of effective energy.
            for option in cand_options.options:
                option.cost -= p_p2p * option.quant_time * option.unit_time

            # Fit the cost model.
            model = ExponentialModel(cand_options)

            # Draw the cost model.
            fig, ax = plt.subplots(figsize=(8, 8), tight_layout=True)
            model.draw(ax, cand_options)
            fig.savefig(f"{output_dir}/{instruction}_{stage}.png")

            # Initialize the operation spec.
            operation = OperationSpec[int](options=cand_options, cost_model=model)
            print(operation.options)
            operation_dict[(instruction, stage)] = operation



def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--inst_profile", type=str, required=True, help="Path for instruction profile results")
    parser.add_argument("--p2p_profile", type=str, help="Path for p2p profile results")
    parser.add_argument("--p2p_power", type=float, help="Raw P2P blocking power consumption value to use")
    parser.add_argument("--output_dir", type=str, required=True, help="Path for output results")
    parser.add_argument("--num_mbs", type=int, default=3, help="Number of microbatchs")
    parser.add_argument("--num_stages", type=int, default=4, help="Number of stages")
    parser.add_argument("--interval", type=int, default=100, help="The interval (number of iterations accumulated) to report pipeline graph and frequency assignment")
    parser.add_argument("--unit_time", type=float, default=0.001, help="The unit of reduction for each iteration, the smaller the value, the more iterations it takes to converge and the finer graunularity for the Pareto frontier")
    parser.add_argument("--fit_method", type=str, default="exponential", choices=["linear", "piecewise-linear", "exponential"], help="Methods to fit the time costs")
    parser.add_argument("--initial_guess", type=str, default="", help="Path for a initial guess of exponential fit parameters")
    parser.add_argument("--replace_time", type=str, help="Path for a file containing the time data to replace the original time in time costs")
    parser.add_argument("--train_schedule", type=str, choices=["1f1b", "early_recomputation_1f1b"], default="1f1b", help="Pipeline schedule.")
    return parser.parse_args()


if __name__ == "__main__":
    main()