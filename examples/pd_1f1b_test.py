import matplotlib.pyplot as plt

from rene import InstructionDAG, Synchronous1F1B, Forward, Backward, PipelineVisualizer


# Instantiate the Instruction DAG.
dag = InstructionDAG(
    schedule_type=Synchronous1F1B,
    num_stages=2,
    num_micro_batches=2,
    # F-0 y=-2x+60 F-1 y=-x+50 B-0 y=-3x+80 B-1 y=-4x+100
    # time_costs={Forward: {0: [(10, 40, 1000), (30, 20, 500), (20, 30, 700)],
    # 1: [(10, 40, 1000), (20, 20, 400), (15, 30, 700)], 2: [(10, 40, 1000), (30, 20, 500), (20, 30, 700)],
    # 3: [(10, 40, 1000), (20, 20, 400), (15, 30, 700)]}, 
    # Backward: {0: [(10, 50, 1000), (20, 20, 500), (15, 35, 700)],
    # 1: [(10, 60, 1000), (20, 20, 400), (15, 40, 700)], 2: [(10, 50, 1000), (20, 20, 500), (15, 35, 700)],
    # 3: [(10, 60, 1000), (20, 20, 400), (15, 40, 700)]}},
    time_costs={Forward: {0: [(10, 40, 1000), (20, 20, 400), (15, 30, 700)],
    1: [(10, 40, 1000), (30, 20, 500), (20, 30, 700)]}, 
    Backward: {0: [(10, 50, 1000), (20, 20, 500), (15, 35, 700)],
    1: [(10, 60, 1000), (20, 20, 400), (15, 40, 700)]}},
)
# Schedule instructions with the "eager" scheduling algorithm.
# Refer to the docstring for available scheduling algorithms.
dag.schedule("pd")

# Pass the DAG to the pipeline visualizer.
# Refer to the constructor for matplotlib customization.
vis = PipelineVisualizer(dag)

# # Instantitate a matplotlib subplot and draw the pipeline and critical path.
fig, ax = plt.subplots(figsize=(12, 4), tight_layout=True)
vis.draw(ax, draw_time_axis=True)
vis.draw_critical_path(ax)
fig.savefig("pipeline.png")
