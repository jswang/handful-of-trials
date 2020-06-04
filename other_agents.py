from datetime import datetime
from dmbrl.config import safety_point_goal_1
import gym, safety_gym
from safe_rl.utils.run_utils import setup_logger_kwargs
from safe_rl.utils.mpi_tools import mpi_fork
from safe_rl import *

env_config = safety_point_goal_1.SafetyPointGoal1ConfigModule()

# copied from experiment.py
num_steps = 2e6
steps_per_epoch = 30000
epochs = int(num_steps // steps_per_epoch)
save_freq = 50
target_kl = 0.01
cost_lim = 25
seed = 42
# make exp name
now = datetime.now()
for algo in ['trpo', 'ppo', 'cpo']:
    exp_name = now.strftime(f"{algo}_%m_%d_%H_%M_%S")
    logger_kwargs = setup_logger_kwargs(exp_name, seed)
    algo = eval(algo)
    algo(
        env_fn = lambda : env_config.get_env(),
        ac_kwargs=dict(
            hidden_sizes=(256, 256),
        ),
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        save_freq=save_freq,
        target_kl=target_kl,
        cost_lim=cost_lim,
        seed=seed,
        logger_kwargs=logger_kwargs
        )