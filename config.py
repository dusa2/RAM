win_size = 16               # The size of glimps
batch_size = 64
loc_std = 0.22
depth = 1
hg_size = hl_size = 128     # The vector size for glimps network (before merge)
g_size = 256                # The size of glimps vector (after merge)
cell_output_size = 256
loc_dim = 2                 # (batch_num, x_coor, y_coor)
cell_size = 256
cell_out_size = cell_size
num_glimpses = 8            # The number of glimps that the paper use to get the lowest loss
num_classes = 3            # MNIST has 10 classes
max_grad_norm = 5.
step = 200               # Epochs
M = 10                      # Monte Carlo sampling
