<h1 align="center">Poise: A Time-Cost Tradeoff Problem Solver</h1>

Poise is a library for solving the [time-cost tradeoff problem](https://link.springer.com/chapter/10.1007/978-3-030-61423-2_5).

## Where do I use Poise?

Say you want to execute a **DAG of tasks**, and each task has multiple execution options with **different time and cost**.

Poise will find the **complete time-cost Pareto frontier** of the entire DAG. Each point on the DAG will be annotated with the right execution option for each task.

You define cost. Any floating point number that is at odds with time.

<!--
## Getting started


## Architecture

**Please note this architecture is outdated and will be updated soon.**

```
              instantiate   ┌───────────────┐    visualize
          ┌────────────────►│  Instruction  │◄────────────────┐
          │                 └───────────────┘                 │
          │                     ▲                             │
          │                     │ schedule                    │
          │                     │                             │
┌─────────┴──────────┐     ┌────┴────────────┐     ┌──────────┴───────────┐
│  PipelineSchedule  ├────►│     ReneDAG     ├────►│  PipelineVisualizer  │
└────────────────────┘     └─────────────────┘     └──────────────────────┘
```

Specify the pipeline schedule (e.g. Synchronous 1F1B) by defining a subclass of `PipelineSchedule`, which is intended to be very similar to DeepSpeed's `PipeSchedule` class.
A `PipelineSchedule` instance defines the order of `Instruction`s executed on each device, and thus instantiates and yields a stream of `Instruction`s that form a *linear chain* of dependencies in the instruction DAG.
See `rene.schedule.Synchronous1F1B` for an example.

`ReneDAG` accepts the pipeline schedule class and pipeline parameters such as the number of stages and microbatches, the duration of each instruction yielded by `PipelineSchedule`, and an optional list of dependency rules, and generates the full list of `Instruction`s by instantiating and invoking the `PipelineSchedule` class passed in.
Refer to `ReneDAG`'s docstring for details on defining custom dependency rules.
You should call `ReneDAG.schedule` in order to assign actual start and finish times for each instruction in the DAG.
Refer to the method's docstring for details on supported scheduling algorithms.

Finally, `PipelineVisualizer` accepts the scheduled `ReneDAG` instance and optional Matplotlib arguments, and generates the final figure.

Also refer to `examples`.
-->
