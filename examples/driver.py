import datetime
import logging
import time
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
plt.rc("svg", hashsalt=None)

from examples.common import df_to_time_costs_pareto, preprocess_time_costs, parse_args
from rene import (
    CriticalDAG,
    Synchronous1F1B,
    PDSolver,
)


def main():
    # Build an argument parser.
    args = parse_args()
        
    inst_profile = args.inst_profile
    p2p_profile = args.p2p_profile
    output_dir = args.output_dir
    num_mbs = args.num_mbs
    
    # Instruction offline profiling results.
    inst_df = pd.read_csv(inst_profile)
    time_costs = df_to_time_costs_pareto(inst_df)
    
    # print(time_costs)

    # P2P communication blocking power consumption.
    p2p_block_df = pd.read_csv(p2p_profile)
    p2p_block_df = p2p_block_df.loc[p2p_block_df.time_ms == 100]
    p2p_block_df["power"] = p2p_block_df.energy_mj / p2p_block_df.time_ms / 100

    # Compute the average power consumption of blocking on P2P communication.
    # In the absolute majority of the times we don't go below 800MHz,
    # so we filter frequencies that are below that and take the average so that we're as accurate as possible.
    p_p2p = p2p_block_df.query("freq >= 800").power.mean().item()

    # time_stamp = datetime.datetime.fromtimestamp(
    #     time.time()).strftime('%m%d_%H%M%S')
    # output_dir = Path(output_dir) / time_stamp
    output_dir = Path(output_dir)
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
    # Instantiate the Instruction DAG.
    dag = CriticalDAG(
        schedule_type=Synchronous1F1B,
        num_stages=4,
        num_micro_batches=num_mbs,
        time_costs=time_costs,  # NOTE: This is from inst_df, not sub_p2p_inst_df, because we want to use the original energy to determine colors.
        output_dir=str(output_dir),
        fit_method=fit_method,
        p2p_power=p_p2p,
    )
    pd_solver = PDSolver(dag, output_dir.__str__(), interval, unit_time)
    pd_solver.run_pd_algorithm()

if __name__ == "__main__":
    main()
