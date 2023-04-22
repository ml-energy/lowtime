import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os
import datetime
import time
import argparse
import pandas as pd
from pathlib import Path
from examples.common import df_to_time_costs_pareto, preprocess_time_costs
from rene  import Forward, Backward

parser = argparse.ArgumentParser()
parser.add_argument("--inst_profile", type=str, required=True, help="Path for instruction profile results")
parser.add_argument("--unit_time", type=float, default=0.001, help="The unit of reduction for each iteration, the smaller the value, the more iterations it takes to converge and the finer graunularity for the Pareto frontier")
parser.add_argument("--p2p_power", type=float, help="Raw P2P blocking power consumption value to use")
parser.add_argument("--num_stages", type=int, default=4, help="Number of stages in the pipeline")
parser.add_argument("--replace_time", type=str, help="Path for a file containing the time data to replace the original time in time costs")
args = parser.parse_args()

inst_df = pd.read_csv(args.inst_profile)
if args.replace_time is not None:
    new_time_df = pd.read_csv(args.replace_time)
    inst_df = inst_df.merge(new_time_df, on=["stage", "instruction", "frequency"], how="left", suffixes=('_x', '_y'))
    inst_df["time_x"] = inst_df["time_y"]
    inst_df = inst_df.drop(columns=["time_y"])
    inst_df = inst_df.rename(columns={"time_x": "time"})
total_time_costs = df_to_time_costs_pareto(inst_df)
total_time_costs = preprocess_time_costs(total_time_costs, args.unit_time)
print(total_time_costs)
   
p2p_power = args.p2p_power

time_stamp = datetime.datetime.fromtimestamp(
        time.time()).strftime('%m%d_%H%M%S')
profile_path = Path(args.inst_profile)
output_dir = Path("/users/yilegu/exponential") / profile_path.name / time_stamp
# output_dir = Path(output_dir)
output_dir.mkdir(parents=True, exist_ok=False)

cands_a = [1e+1, 1e+2, 1e+3]
cands_b = np.array([-0.01, -0.02, -0.03, -0.04, -0.05, -0.1])
cands_b *= (args.unit_time / 0.001)  
cands_b = list(cands_b)
cands_c = [1e+1, 1e+2, 1e+3]

for inst_type in [Forward, Backward]:
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
                    fit_coeffs, _ = curve_fit(
                        lambda t, a, b, c: a * np.exp(b * t) + c,
                        time_list,
                        cost_list,
                        p0=p0,
                        maxfev=10000,
                    ) 
                    fig, ax = plt.subplots(figsize=(8, 8), tight_layout=True)
                    ax.plot(time_list, cost_list, "o")
                    for i in range(len(time_list)):
                        ax.annotate(
                            f"({time_list[i]:.6f}, {cost_list[i]:.6f}, {freq_list[i]})",
                            (time_list[i], cost_list[i]),
                        )
                    # ax.plot(time_list, cost_list_unrefined, "x")
                    # generate a list with step size 0.1
                    x = np.arange(min(time_list), max(time_list), 0.0001)
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
                        os.path.join(str(output_dir), f"exopnential_{inst_type}_{stage_id}_{p0}.png"), format="PNG"
                    )
                    plt.clf()
                    plt.close()