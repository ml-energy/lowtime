import matplotlib.pyplot as plt

from rene import InstructionDAG, Synchronous1F1B, Forward, Backward, PipelineVisualizer


# Instantiate the Instruction DAG.
dag = InstructionDAG(
    schedule_type=Synchronous1F1B,
    num_stages=5,
    num_micro_batches=5,
    durations={Forward: [1.0, 1.0, 1.4, 1.0, 1.2], Backward: [1.5, 1.5, 2.5, 1.5, 1.6]},
)
# Schedule instructions with the "eager" scheduling algorithm.
# Refer to the docstring for available scheduling algorithms.
dag.schedule("eager")

# Pass the DAG to the pipeline visualizer.
# Refer to the constructor for matplotlib customization.
vis = PipelineVisualizer(dag)

# Instantitate a matplotlib subplot and draw the pipeline and critical path.
fig, ax = plt.subplots(figsize=(12, 4), tight_layout=True)
vis.draw(ax, draw_time_axis=True)
vis.draw_critical_path(ax)
fig.savefig("pipeline.png")
