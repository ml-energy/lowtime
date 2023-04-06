import argparse
import datetime
import logging
import shutil
import sys
import time
import yaml
from typing import Type
from matplotlib.patches import Patch
from matplotlib.colors import Normalize
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc("svg", hashsalt=None)

from examples.common import df_to_time_cost_pareto, subtract_p2p
from rene import (
    CriticalDAG,
    Synchronous1F1B,
    PDSolver,
)


def main():
    if len(sys.argv) != 2:
        print("Usage: ./driver.py [YAML_PATH]")
        return
    
    with open(sys.argv[1]) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
        
    general_conf = conf["general"]
    inst_profile = general_conf["inst_profile"]
    p2p_profile = general_conf["p2p_profile"]
    output_dir = general_conf["output_dir"]
    num_mbs = general_conf["num_mbs"]
    


    # Instruction offline profiling results.
    # inst_df = pd.read_csv("/users/yilegu/perseus-analysis/data/merak_offline_profiler/merak+bert-large-uncased+uniform_transformer+dp1+tp1+pp4+mbs16.csv")
    inst_df = pd.read_csv(inst_profile)
    time_costs = df_to_time_cost_pareto(inst_df)

    print(time_costs)

    # P2P communication blocking power consumption.
    # p2p_block_df = pd.read_csv("../perseus-analysis/data/p2p-benchmark/intranode-bare-nvlink-sleep-1665-0-1.csv")
    p2p_block_df = pd.read_csv(p2p_profile)
    p2p_block_df = p2p_block_df.loc[p2p_block_df.time_ms == 100]
    p2p_block_df["power"] = p2p_block_df.energy_mj / p2p_block_df.time_ms / 100

    # Compute the average power consumption of blocking on P2P communication.
    # In the absolute majority of the times we don't go below 800MHz,
    # so we filter frequencies that are below that and take the average so that we're as accurate as possible.
    p_p2p = p2p_block_df.query("freq >= 800").power.mean().item()

    # def subtract_p2p(row):
    #         row.energy -= row.time * p_p2p
    #         return row
    # inst_df = inst_df.apply(subtract_p2p, axis=1)

    time_stamp = datetime.datetime.fromtimestamp(
        time.time()).strftime('%m%d_%H%M%S')
    output_dir = Path.joinpath(Path(output_dir), time_stamp)
    output_dir.mkdir(parents=True, exist_ok=False)

    log_path = Path.joinpath(output_dir, 'job.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='(%m-%d) %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_path, mode='a'),
            logging.StreamHandler()
        ])
    
    # Algorithm part
    algo_conf = conf["algorithm"]
    if algo_conf["type"] == "pd":
        interval = algo_conf["interval"]
        unit_scale = algo_conf["unit_scale"]
        fit_method = algo_conf["fit_method"]
        # Quantize the time costs and preprocess data points: remove redundant time points, break ties by lower cost value.
        for stage_to_time_costs in time_costs.values():
            for stage, time_cost_list in stage_to_time_costs.items():
                time_cost_list = [(t // unit_scale * unit_scale, e, f) for t, e, f in time_cost_list]
                # turn the 3-tuple list into 3D numpy array, use float for frequency as numpy array needs to have the same type
                time_cost_array = np.asarray(time_cost_list, dtype=[('time', float), ('cost', float), ('freq', float)])
                # Sort the points by their x-coordinate in ascending order, break ties by choosing the point with the smallest y-coordinate
                time_cost_array = time_cost_array[time_cost_array.argsort(order=["time", "cost"])]
                # time_cost_array = time_cost_array.reshape(time_cost_array.shape[0], 3)
                time_cost_array = time_cost_array.view((float, 3))
                # time_cost_array = time_cost_array.view(dtype=[('time', float), ('cost', float), ('freq', int)], type=np.ndarray)
                # Remove duplicate points by x-coordinate, break ties by choosing the point with the smallest y-coordinate
                time_cost_array = time_cost_array[np.unique(time_cost_array[:, 0], return_index=True)[1]]
                # Retrive the new time_cost_list
                time_cost_list = list(zip(time_cost_array[:, 0], time_cost_array[:, 1], time_cost_array[:, 2].astype(int)))
                stage_to_time_costs[stage] = time_cost_list


        # Instantiate the Instruction DAG.
        dag = CriticalDAG(
            schedule_type=Synchronous1F1B,
            num_stages=4,
            num_micro_batches=num_mbs,
            time_costs=time_costs,  # NOTE: This is from inst_df, not sub_p2p_inst_df, because we want to use the original energy to determine colors.
            output_dir=output_dir.__str__(),
            fit_method=fit_method,
            p2p_power=p_p2p,
        )
        pd_solver = PDSolver(dag, output_dir.__str__(), interval, unit_scale)
        pd_solver.run_pd_algorithm()
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
