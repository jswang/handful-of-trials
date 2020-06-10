#!/bin/bash

# Make sure virtual env is activated
# source hand_venv/bin/activate

# Vanilla pets on cartpole
if [ "$1" = "cartpole" ]; then
    python scripts/mbexp.py -env cartpole
# Vanilla pets on safety
elif [ "$1" = "safety_point_goal_1" ]; then
    python scripts/mbexp.py -env safety_point_goal_1
# PE-TS with more pretraining for any network
elif [ "$1" = "pretrain" ]; then
    python scripts/mbexp.py -env "$2" -o exp_cfg.exp_cfg.ninit_rollouts 100 -o ctrl_cfg.prop_cfg.model_train_cfg.initial_epochs 35 -o ctrl_cfg.prop_cfg.model_train_cfg.holdout_ratio .2  -o ctrl_cfg.prop_cfg.model_init_cfg.model_dir "data/train_val/$2"
# Fast run Cartpole task horizon=5 (i/o 200), MPC planning horizon=5 (i/o 25)
elif [ "$1" = "fast" ]; then
    python scripts/mbexp.py -env "$2" -o exp_cfg.exp_cfg.ntrain_iters 1 -o exp_cfg.sim_cfg.task_hor 1 -o ctrl_cfg.opt_cfg.plan_hor 1 -logdir "data/tmp"

# Safeopt
elif [ "$1" = "safeopt" ]; then
    python3 scripts/mbexp.py -env "$2" -ca opt-type SafeOpt

# Run PPO, TRPO, and CPO on safety env
elif [ "$1" = "run_agents" ]; then
    python test_other_agents.py --algo 'trpo'
    python test_other_agents.py --algo 'ppo'
    python test_other_agents.py --algo 'cpo'
elif [ "$1" = "plot" ]; then
    python plot.py data/2020-06-07_trpo_PointSafety/ --savedir 'data/other_agents' --title 'TRPO Safety Point Goal 1' --dont_show
    python plot.py data/2020-06-07_cpo_PointSafety/ --savedir 'data/other_agents' --title 'CPO Safety Point Goal 1' --dont_show
    python plot.py data/2020-06-07sour_ppo_PointSafety/ --savedir 'data/other_agents' --title 'PPO Safety Point Goal 1' --dont_show
else
    echo "Options: plot, run_agents, safety_point_goal_1, cartpole, pretrain <env>, fast <env>"
fi

# Default settings
# {'ctrl_cfg': {'env': <dmbrl.env.cartpole.CartpoleEnv object at 0x7fae5fe8d470>,
#               'opt_cfg': {'ac_cost_fn': <function CartpoleConfigModule.ac_cost_fn at 0x7fae6010b840>,
#                           'cfg': {'alpha': 0.1,
#                                   'max_iters': 5,
#                                   'num_elites': 40,
#                                   'popsize': 400},
#                           'mode': 'CEM',
#                           'obs_cost_fn': <function CartpoleConfigModule.obs_cost_fn at 0x7fae6010b7b8>,
#                           'plan_hor': 25},
#               'prop_cfg': {'mode': 'TSinf',
#                            'model_init_cfg': {'model_class': <class 'dmbrl.modeling.models.BNN.BNN'>,
#                                               'model_constructor': <bound method CartpoleConfigModule.nn_constructor of <cartpole.CartpoleConfigModule object at 0x7fae5fe8d0b8>>,
#                                               'num_nets': 5},
#                            'model_train_cfg': {'epochs': 5},
#                            'npart': 20,
#                            'obs_postproc': <function CartpoleConfigModule.obs_postproc at 0x7fae6010b6a8>,
#                            'obs_preproc': <function CartpoleConfigModule.obs_preproc at 0x7fae6010b620>,
#                            'targ_proc': <function CartpoleConfigModule.targ_proc at 0x7fae6010b730>}},
#  'exp_cfg': {'exp_cfg': {'nrollouts_per_iter': 1, 'ntrain_iters': 50},
#              'log_cfg': {'logdir': 'log'},
#              'sim_cfg': {'env': <dmbrl.env.cartpole.CartpoleEnv object at 0x7fae5fe8d470>,
#                          'task_hor': 200}}}
