import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os
import datetime
import time
import argparse
import pandas as pd
from pathlib import Path
from examples.common import df_to_time_costs_pareto, preprocess_time_costs, offline_df_to_recomputation_df
from rene  import Forward, Backward, ForwardBackward

parser = argparse.ArgumentParser()
parser.add_argument("--inst_profile", type=str, required=True, help="Path for instruction profile results")
parser.add_argument("--unit_time", type=float, default=0.001, help="The unit of reduction for each iteration, the smaller the value, the more iterations it takes to converge and the finer graunularity for the Pareto frontier")
parser.add_argument("--p2p_power", type=float, help="Raw P2P blocking power consumption value to use")
parser.add_argument("--num_stages", type=int, default=4, help="Number of stages in the pipeline")
parser.add_argument("--replace_time", type=str, help="Path for a file containing the time data to replace the original time in time costs")
parser.add_argument("--train_schedule", type=str, choices=["1f1b", "early_recomputation_1f1b"], default="1f1b", help="Pipeline schedule.")
args = parser.parse_args()

inst_df = pd.read_csv(args.inst_profile)
if args.replace_time is not None:
    new_time_df = pd.read_csv(args.replace_time)
    inst_df = inst_df.merge(new_time_df, on=["stage", "instruction", "frequency"], how="left", suffixes=('_x', '_y'))
    inst_df["time_x"] = inst_df["time_y"]
    inst_df = inst_df.drop(columns=["time_y"])
    inst_df = inst_df.rename(columns={"time_x": "time"})
total_time_costs = df_to_time_costs_pareto(inst_df)
if args.train_schedule == "early_recomputation_1f1b":
    augmented_time_costs = total_time_costs.copy()
    augmented_time_costs[ForwardBackward] = total_time_costs[Backward]
    new_df = offline_df_to_recomputation_df(inst_df)
    augmented_time_costs[Backward] = df_to_time_costs_pareto(new_df)[Backward] 
    total_time_costs = augmented_time_costs
total_time_costs = preprocess_time_costs(total_time_costs, args.unit_time)
print(total_time_costs)
   
p2p_power = args.p2p_power

time_stamp = datetime.datetime.fromtimestamp(
        time.time()).strftime('%m%d_%H%M%S')
profile_path = Path(args.inst_profile)
output_dir = Path("./results/exponential") / profile_path.name / time_stamp
# output_dir = Path(output_dir)
output_dir.mkdir(parents=True, exist_ok=False)

cands_a = [1e+1, 1e+2, 1e+3]
cands_b = np.array([-0.01, -0.02, -0.03, -0.04, -0.05, -0.1])
cands_b *= (args.unit_time / 0.001)  
cands_b = list(cands_b)
cands_c = [1e+1, 1e+2, 1e+3]

types = [Forward, Backward, ForwardBackward] if args.train_schedule == "early_recomputation_1f1b" else [Forward, Backward]

for inst_type in types:
    for stage_id in range(args.num_stages):
        time_costs = total_time_costs[inst_type][stage_id]
        time_costs.sort(key=lambda x: x[0], reverse=True)
        time_list = []
        cost_list = []
        freq_list = []
        cost_list_unrefined = []
        for t, e, f in time_costs:
            time_list.append(t)
            cost_list_unrefined.append(e)
            cost_list.append(e - p2p_power * t * args.unit_time)
            freq_list.append(f)
        for i, cand_a in enumerate(cands_a):
            for j, cand_b in enumerate(cands_b):
                for k, cand_c in enumerate(cands_c):
                    p0 = [cand_a, cand_b, cand_c]
                    fit_coeffs, pcov = curve_fit(
                        lambda t, a, b, c: a * np.exp(b * t) + c,
                        time_list,
                        cost_list,
                        p0=p0,
                        maxfev=10000,
                    ) 

                    if np.inf in pcov:
                        l2_err = np.inf
                    else:
                        a, b, c = fit_coeffs
                        preds = [a * np.exp(b * t) + c for t in time_list]
                        l2_err = np.mean(np.square(np.array(preds) - np.array(cost_list)))

                    fig, ax = plt.subplots(figsize=(8, 8), tight_layout=True)
                    ax.plot(time_list, cost_list, "o")
                    for i in range(len(time_list)):
                        ax.annotate(
                            f"({time_list[i]:d}, {cost_list[i]:.6f}, {freq_list[i]})",
                            (time_list[i], cost_list[i]),
                        )
                    # ax.plot(time_list, cost_list_unrefined, "x")
                    # generate a list with step size 0.1
                    x = np.arange(min(time_list), max(time_list), 0.01)
                    # ax.plot(x, np.polyval(self.fit_coeffs, x), 'r-')
                    y = []
                    for m in x:
                        y.append(fit_coeffs[0] * np.exp(fit_coeffs[1] * m) + fit_coeffs[2])
                    # a, b, c = unrefined_fit_coeffs
                    # unrefined_y = a * np.exp(b * x) + 
                    # refined_y = []
                    # for i in x:
                    #     refined_y.append(self.get_p2p_refined_cost(i))
                    ax.plot(x, y, "r-")
                    ax.set_xlabel("time")
                    ax.set_ylabel("energy")
                    fig.savefig(
                        os.path.join(str(output_dir), f"exopnential_{inst_type.__name__}_{stage_id}_{p0}_{fit_coeffs}_{l2_err}.png"), format="PNG"
                    )
                    plt.clf()
                    plt.close()