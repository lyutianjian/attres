# TinyStories baseline tuned for RTX5090 32G high-throughput runs.

out_dir = 'out-tinystories-12l'
eval_interval = 400
eval_iters = 100
log_interval = 10

always_save_checkpoint = False

wandb_log = True
wandb_project = 'tinystories'
wandb_run_name = 'baseline-12l-320d-rtx5090-bs64-b512-cmp'
track_diagnostics = True
diagnostics_interval = 100

dataset = 'tinystories'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 512

# favor depth over width so AttnRes has room to help later
n_layer = 12
n_head = 8
n_embd = 320
dropout = 0.0
use_attn_res = False

learning_rate = 3e-4
max_iters = 20000
lr_decay_iters = 20000
min_lr = 3e-5
beta2 = 0.99
warmup_iters = 200

# RTX5090 speed profile
compile = True
