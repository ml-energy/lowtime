import argparse
import datetime
import time
import multiprocessing
import subprocess
from pathlib import Path
import pandas as pd

def run_task(task: dict):
    # run the driver script
    arg_list = []
    for key, value in task.items():
        if key == "task_name" or key == "driver_path":
            continue
        arg_list.append(f"--{key}")
        arg_list.append(str(value))
    subprocess.run(["python", task["driver_path"]] + arg_list, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return task["task_name"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload_path", type=str, required=True, help="Path for workload file")
    parser.add_argument("--data_path", type=str, required=True, help="Path for directory containing instruction profile results")
    parser.add_argument("--p2p_profile", type=str, help="Path for p2p profile results")
    parser.add_argument("--p2p_power", type=float, help="Raw P2P blocking power consumption value to use")
    parser.add_argument("--unit_time", type=float, help="The unit time for PD algorithm")
    parser.add_argument("--output_dir", type=str, required=True, help="Path for output results, this will contain the results of each task")
    parser.add_argument("--driver_path", type=str, required=True, help="Path for driver script")
    args = parser.parse_args()
    # sanity check p2p profile
    if args.p2p_profile is not None:
        assert(Path(args.p2p_profile).exists())
    assert(Path(args.driver_path).exists())
    # import csv dataframe from workload path
    df = pd.read_csv(args.workload_path)

    time_stamp = datetime.datetime.fromtimestamp(
        time.time()).strftime('%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / time_stamp
    output_dir.mkdir(parents=True, exist_ok=False)

    tasks = []
    # assume merak is the framework
    framework = "merak"
    # iterate through each row in the dataframe, create a task list
    for index, row in df.iterrows():
        base_args = {}
        # first locate the inst profile
        base_name = f"{framework}+{row['model']}+{row['partition_method']}+dp{row['dp']}+pp{row['pp']}+tp{row['tp']}+mbs{row['microbatch_size']}"
        inst_profile = base_name + ".csv"
        inst_profile = Path(args.data_path) / inst_profile
        print(inst_profile)
        # sanity check
        assert(inst_profile.exists())
        # create task args, use the same base for all 3 fit methods
        base_args["driver_path"] = args.driver_path
        base_args["inst_profile"] = str(inst_profile)
        base_args["p2p_profile"] = args.p2p_profile
        base_args["num_mbs"] = row["num_microbatches"]
        # use default values for interval and unit_time
        base_args["interval"] = 500
        base_args["unit_time"] = args.unit_time
        for fit_method in ["linear", "piecewise-linear", "exponential"]:
            task_args = base_args.copy()
            if fit_method == "exponential":
                initial_guess = base_name + "+exponential.py"
                initial_guess = Path(args.data_path) / initial_guess
                if initial_guess.exists():
                    task_args["initial_guess"] = initial_guess
            task_args["fit_method"] = fit_method
            # task_name = base_name + f"+{fit_method}"
            task_args["task_name"] = base_name
            task_output_dir = str(output_dir / fit_method / base_name) + f"+nmb{row['num_microbatches']}"
            task_args["output_dir"] = task_output_dir
            tasks.append(task_args)

    with multiprocessing.Pool() as p:
        res_itr = p.imap_unordered(run_task, tasks)
        for i, res in enumerate(res_itr):
            print(f'Completed {i+1}/{len(tasks)}: {res}', end='\r')
        print('\n')

if __name__ == "__main__":
    main()
