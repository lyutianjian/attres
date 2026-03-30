# Full AttnRes counterpart to train_tinystories.py for clean A/B experiments.

out_dir = 'out-tinystories-12l-attnres'
eval_interval = 400
eval_iters = 100
log_interval = 10

always_save_checkpoint = False

wandb_log = True
wandb_project = 'tinystories'
wandb_run_name = 'attnres-12l-320d'
track_diagnostics = True
diagnostics_interval = 100

dataset = 'tinystories'
gradient_accumulation_steps = 8
batch_size = 8
block_size = 256

n_layer = 12
n_head = 8
n_embd = 320
dropout = 0.0

use_attn_res = True
attn_res_rms_eps = 1e-5

learning_rate = 3e-4
max_iters = 20000
lr_decay_iters = 20000
min_lr = 3e-5
beta2 = 0.99
warmup_iters = 200

compile = False
