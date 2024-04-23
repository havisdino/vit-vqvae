# Data config
img_size = (128, 128)
patch_size = (8, 8)
strides = (8, 8)

# Model config
d_model = 256
d_patch = 192
codebook_size = 512
dff = 512
seqlen = 256
n_heads = 4
n_blocks = 5
dropout = 0.1

# Training config
warmup_step = 180
init_lr = 3e-3
peak_lr = 3e-3
min_lr = 5e-4
down_weight = 80 # the less the steeper