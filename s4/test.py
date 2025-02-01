import os

run_id = "checkpoints\mnist\s4-d_model=128-lr=0.001-bsz=128\checkpoint_0.orbax-checkpoint-tmp-0"

abs_path = os.path.join(os.getcwd(), run_id)

print(abs_path)
