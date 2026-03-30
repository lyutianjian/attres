# Deep-and-narrow TinyStories baseline to stress residual aggregation on a single 8GB GPU.

out_dir = 'out-tinystories-20l'
eval_interval = 400
eval_iters = 100
log_interval = 10

always_save_checkpoint = False

wandb_log = True
wandb_project = 'tinystories'
wandb_run_name = 'baseline-20l-256d'
track_diagnostics = True
diagnostics_interval = 100

dataset = 'tinystories'
gradient_accumulation_steps = 16
batch_size = 4
block_size = 256

n_layer = 20
n_head = 8
n_embd = 256
dropout = 0.0
use_attn_res = False

learning_rate = 3e-4
max_iters = 28000
lr_decay_iters = 28000
min_lr = 3e-5
beta2 = 0.99
warmup_iters = 400

compile = False
