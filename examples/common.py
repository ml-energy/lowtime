from typing import Type
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from rene import (
    Instruction,
    Forward,
    Backward,
)

TIME_COST_T = dict[Type[Instruction], dict[int, list[tuple[float, float, int]]]]

def df_to_time_costs_pareto(inst_df: pd.DataFrame) -> TIME_COST_T:
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

def preprocess_time_costs(time_costs: TIME_COST_T, unit_time: float) -> TIME_COST_T:
    """Preprocess time costs data. Quantize the time costs and preprocess data points: remove redundant time points, break ties by lower cost value.
    
    Arguments:
        time_costs: time costs data
        unit_time: time unit to quantize the time costs

    Returns:
        Preprocessed time costs data
    """
    processed_time_costs: TIME_COST_T = time_costs.copy()
    for stage_to_time_costs in processed_time_costs.values():
        for stage, time_cost_list in stage_to_time_costs.items():
            time_cost_list = [(t // unit_time * unit_time, e, f) for t, e, f in time_cost_list]
            # Turn the 3-tuple list into 3D numpy array, use float for frequency as numpy array needs to have the same type
            time_cost_array = np.asarray(time_cost_list, dtype=[('time', float), ('cost', float), ('freq', float)])
            # Sort the points by their x-coordinate in ascending order, break ties by choosing the point with the smallest y-coordinate
            time_cost_array = time_cost_array[time_cost_array.argsort(order=["time", "cost"])]
            time_cost_array = time_cost_array.view((float, 3))
            # Remove duplicate points by x-coordinate, break ties by choosing the point with the smallest y-coordinate
            time_cost_array = time_cost_array[np.unique(time_cost_array[:, 0], return_index=True)[1]]
            # Retrive the new time_cost_list
            time_cost_list = list(zip(time_cost_array[:, 0], time_cost_array[:, 1], time_cost_array[:, 2].astype(int)))
            stage_to_time_costs[stage] = time_cost_list
    return processed_time_costs