from __future__ import annotations
import itertools

import logging
import argparse
from pathlib import Path
from typing import Literal, Optional, Union, Type
from collections import defaultdict
from dataclasses import dataclass

import tyro
import pandas as pd
from matplotlib import pyplot as plt

from rene.operation import (
    CandidateExecutionOptions,
    OperationSpec,
    ExecutionOption,
    ExponentialModel,
)
from rene.perseus.instruction import (
    Instruction,
    Forward,
    Backward,
    forward_dep,
    backward_dep,
)
from rene.perseus.schedule import Synchronous1F1B
from rene.dag import DependencyResolver, ReneDAG

logger = logging.getLogger()


@dataclass
class Args:
    # Path to instruction profile results
    inst_profile: str
    # Raw P2P blocking power consumption, in Watts
    p2p_power: Union[float, str]
    # Directory to output results
    output_dir: str
    # Number of microbatchs
    num_mbs: int
    # Number of stages
    num_stages: int
    # Interval to output heavy results like plots
    interval: int = 100
    # The unit of reduction for each iteration, in seconds
    unit_time: float = 0.001
    # Methods to fit the time costs
    fit_method: Literal["linear", "piecewise-linear", "exponential"] = "exponential"
    # Path to exponential model initial parameter guesses
    initial_guess: Optional[str] = None
    # Path for a file containing the time data to replace the time in --inst_profile
    replace_time: Optional[str] = None
    # Pipeline schedule name
    train_schedule: Literal["1f1b", "early_recomputation_1f1b"] = "1f1b"


def main(
    args: Args
) -> None:
    """Perseus time-cost tradeoff optimization."""
    # Validate arguments.
    if args.train_schedule != "1f1b":
        raise NotImplementedError("Only 1f1b is supported for now.")
    if args.fit_method != "exponential":
        raise NotImplementedError("Only exponential is supported for now.")

    # Setup logging and output.
    output_dir= Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    log_path = output_dir/ "job.log"

    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.FileHandler(log_path, mode="a"), logging.StreamHandler()],
    )
    logger.info("Arguments: %s", args)

    # Instruction offline profiling results.
    inst_df = pd.read_csv(args.inst_profile)

    # If specified, merge in online-profiled execution time data.
    if args.replace_time is not None:
        new_time_df = pd.read_csv(args.replace_time)
        # get smallest freq in new_time_df, this will be the cutoff for the inst_df
        min_freq = new_time_df.frequency.min()
        inst_df = inst_df[inst_df.frequency >= min_freq]
        inst_df = inst_df.merge(
            new_time_df,
            on=["stage", "instruction", "frequency"],
            how="left",
            suffixes=("_x", "_y"),
        )
        inst_df["time_x"] = inst_df["time_y"]
        inst_df = inst_df.drop(columns=["time_y"])
        inst_df = inst_df.rename(columns={"time_x": "time"})

    # P2P communication blocking power consumption.
    if isinstance(args.p2p_power, str):
        p2p_block_df = pd.read_csv(args.p2p_power)
        p2p_block_df = p2p_block_df.loc[p2p_block_df.time_ms == 100]
        p2p_block_df["power"] = p2p_block_df.energy_mj / p2p_block_df.time_ms / 100

        # Compute the average power consumption of blocking on P2P communication.
        # In the absolute majority of the times we don't go below 800MHz,
        # so we filter frequencies that are below that and take the average so that we're as accurate as possible.
        p_p2p: float = p2p_block_df.query("freq >= 800").power.mean().item()
    else:
        p_p2p = args.p2p_power
    logger.info("P2P blocking power consumption: %.2fW", p_p2p)

    op_spec_map: dict[int, dict[Type[Instruction], OperationSpec]] = defaultdict(dict)
    for instruction in [Forward, Backward]:
        inst_name = instruction.__name__.lower()
        for stage_id in range(args.num_stages):
            logger.info("Processing %s stage %d", inst_name, stage_id)
            options = []
            _df = inst_df.query("stage == @stage_id and instruction == @instruction")
            for _, row in _df.iterrows():
                options.append(
                    ExecutionOption[int](
                        real_time=row["time"],
                        unit_time=args.unit_time,
                        cost=row["energy"],
                        knob=int(row["frequency"]),
                    )
                )

            # Get the Preto frontier, quantize time, and deduplicate time.
            cand_options = CandidateExecutionOptions[int](options=options)

            # Map the cost to be effective computation energy.
            # Everything from now on is in terms of effective energy.
            for option in cand_options.options:
                option.cost -= p_p2p * option.quant_time * option.unit_time

            # Fit the cost model.
            model = ExponentialModel(cand_options)

            # Draw the cost model.
            fig, ax = plt.subplots(figsize=(8, 8), tight_layout=True)
            model.draw(ax, cand_options)
            fig.savefig(f"{output_dir}/{instruction}_{stage_id}.png")

            # Initialize the operation spec.
            op_spec = OperationSpec[int](options=cand_options, cost_model=model)
            print(op_spec.options)
            op_spec_map[stage_id][instruction] = op_spec

    ########################
    # ReneDAG construction #
    ########################
    dag = ReneDAG[Instruction, None]()

    # Generate and add all instructions to the DAG.
    # Reserve 0 for dummy source and 1 for dummy sink.
    # XXX(JW): Why should all these have a node ID exposed outside?
    node_id = 2
    for stage_id in range(args.num_stages):
        # Generate instructions for each stage.
        stage_node_ids: list[int] = []
        for inst in Synchronous1F1B(
            num_stages=args.num_stages,
            num_micro_batches=args.num_mbs,
            stage_id=stage_id,
            operation_spec_map=op_spec_map[stage_id],
        ):
            dag.add_node(node_id, inst)
            stage_node_ids.append(node_id)
            node_id += 1
        
        # Add dependencies between adjacent instructions in the same stage.
        for node_id1, node_id2 in zip(stage_node_ids, stage_node_ids[1:]):
            dag.add_edge(node_id1, node_id2, None)

    # Add dependencies between dependent pipeline instructions.
    insts = dag.node_attrs.items()
    resolver = DependencyResolver(
        dependency_rules=[forward_dep, backward_dep],
        node_type=Instruction,
    )
    for (id1, inst1), (id2, inst2) in itertools.product(insts, insts):
        if resolver.is_dependent(inst1, inst2):
            dag.add_edge(id1, id2, None)


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)