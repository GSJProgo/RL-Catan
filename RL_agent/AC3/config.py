# Random seed for reproducibility
RANDOM_SEED = 2

# Number of global iterations before updating the global network
UPDATE_GLOBAL_ITER = 10000

# Discount factor for future rewards in the reinforcement learning algorithm
GAMMA = 0.99

# Maximum number of episodes for training
MAX_EP = 500000

# Total number of possible actions in the environment
TOTAL_ACTIONS = 21*11*4 + 41

# Number of GPUs to use for training
NUM_GPUS = 2

# Number of workers per GPU
NUM_WORKERS_PER_GPU = 2

# Learning rate for the optimizer
LR_START = 8e-4

# Decay rate for the learning rate
LR_DECAY_RATE_1 = 0.9999
LR_UPDATE_TRESHOLD_1 = 1e-4
LR_DECAY_RATE_2 = 0.99998
LR_UPDATE_TRESHOLD_2 = 1e-5
LR_DECAY_RATE_3 = 0.999998

# Beta values for the optimizer
OPT_BETA = (0.9, 0.999)

# Weight decay for the optimizer
WEIGHT_DECAY = 0.0001

# Rate at which the model is saved during training
MODEL_SAVE_RATE = 1000

# Threshold for changing the opponent agent during training
OPPONENT_AGENT_CHANGE_THRESHOLD = 4000

# Number of residual blocks for the actor network
NUM_RES_BLOCKS_ACTOR = 6

# Number of residual blocks for the critic network
NUM_RES_BLOCKS_CRITIC = 4

# Loss factor for the critic loss
C_LOSS_FACTOR = 1

# Loss factor for the actor loss
A_LOSS_FACTOR = 1

# Entropy factor for the actor loss
ENTROPY_FACTOR = 1e-4

# L2 activity factor for the actor and critic networks
L2_ACTIVITY_FACTOR = 5e-5
