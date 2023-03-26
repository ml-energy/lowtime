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

import pandas as pd
import matplotlib.pyplot as plt
plt.rc("svg", hashsalt=None)

from examples.common import df_to_time_cost_pareto, subtract_p2p
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
    sub_p2p_inst_df = subtract_p2p(inst_df, p2p_block_df)

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
        # Instantiate the Instruction DAG.
        dag = CriticalDAG(
            schedule_type=Synchronous1F1B,
            num_stages=4,
            num_micro_batches=num_mbs,
            time_costs=time_costs,  # NOTE: This is from inst_df, not sub_p2p_inst_df, because we want to use the original energy to determine colors.
            output_dir=output_dir.__str__(),
            fit_method=fit_method,
        )
        pd_solver = PD_Solver(dag, output_dir.__str__(), interval, unit_scale)
        pd_solver.run_pd_algorithm()
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
