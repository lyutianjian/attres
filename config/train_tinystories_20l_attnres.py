# Deep-and-narrow TinyStories AttnRes tuned for RTX5090 32G high-throughput runs.

out_dir = 'out-tinystories-20l-attnres'
eval_interval = 400
eval_iters = 100
log_interval = 10

always_save_checkpoint = False

wandb_log = True
wandb_project = 'tinystories'
wandb_run_name = 'attnres-20l-256d-rtx5090-bs32-b512-cmp'
track_diagnostics = True
diagnostics_interval = 100

dataset = 'tinystories'
gradient_accumulation_steps = 2
batch_size = 32
block_size = 512

n_layer = 20
n_head = 8
n_embd = 256
dropout = 0.0
use_attn_res = True
attn_res_rms_eps = 1e-5

learning_rate = 3e-4
max_iters = 28000
lr_decay_iters = 28000
min_lr = 3e-5
beta2 = 0.99
warmup_iters = 400

compile = True
