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
    subprocess.run(["python", task["driver_path"]] + arg_list)
    return task["task_name"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload_path", type=str, help="Path for workload file")
    parser.add_argument("--data_path", type=str, help="Path for directory containing instruction profile results")
    parser.add_argument("--profile_list", type=str, help="Path for a list of profiles to run ")
    parser.add_argument("--p2p_power", type=float, help="Raw P2P blocking power consumption value to use")
    parser.add_argument("--unit_time", type=float, help="The unit time for PD algorithm")
    parser.add_argument("--driver_path", type=str, required=True, help="Path for driver script")
    parser.add_argument("--num_stages", type=int, default=4, help="Number of stages in the pipeline")
    parser.add_argument("--replace_time", type=str, help="Path for a file containing the time data to replace the original time in time costs")
    args = parser.parse_args()
    # sanity check input path
    if args.profile_list is None and args.workload_path is None and args.data_path is None or (args.workload_path is not None and args.data_path is None) or (args.workload_path is None and args.data_path is not None):
        raise ValueError("Either profile_list or (workload_path and data_path) must be specified")
    # sanity check p2p profile
    assert(Path(args.driver_path).exists())
    if args.profile_list is None:
        # import csv dataframe from workload path
        df = pd.read_csv(args.workload_path)

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
            # use default values for interval and unit_time
            base_args["unit_time"] = args.unit_time
            base_args["p2p_power"] = args.p2p_power
            base_args["task_name"] = base_name
            base_args["num_stages"] = args.num_stages
            if args.replace_time is not None:
                assert(Path(args.replace_time).exists())
                base_args["replace_time"] = args.replace_time
            tasks.append(base_args)
    else:
        tasks = []
        with open(args.profile_list, "r") as f:
            for line in f:
                inst_profile, _ = line.strip().split(" ")
                base_args = {}
                base_args["driver_path"] = args.driver_path
                base_args["inst_profile"] = inst_profile
                base_args["unit_time"] = args.unit_time
                base_args["p2p_power"] = args.p2p_power
                base_args["task_name"] = Path(inst_profile).stem
                base_args["num_stages"] = args.num_stages
                if args.replace_time is not None:
                    assert(Path(args.replace_time).exists())
                    base_args["replace_time"] = args.replace_time
                tasks.append(base_args)

    with multiprocessing.Pool() as p:
        res_itr = p.imap_unordered(run_task, tasks)
        for i, res in enumerate(res_itr):
            print(f'Completed {i+1}/{len(tasks)}: {res}', end='\r')
        print('\n')

if __name__ == "__main__":
    main()
