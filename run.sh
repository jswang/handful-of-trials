#!/bin/bash

# Make sure virtual env is activated
source hand_venv/bin/activate

# Vanilla pets on cartpole
if [ "$1" = "cartpole" ]; then
    python scripts/mbexp.py -env cartpole
# Vanilla pets on safety1
elif [ "$1" = "safety" ]; then
    python scripts/mbexp.py -env safety_point_goal_1
# PE-TS with more pretraining of the neural network for safety network
elif [ "$1" = "safety_pretrain" ]; then
    python scripts/mbexp.py -env safety_point_goal_1 -o exp_cfg.exp_cfg.ninit_rollouts 5 -o ctrl_cfg.prop_cfg.model_train_cfg.epochs 10 -o ctrl_cfg.prop_cfg.model_train_cfg
# Fast run Cartpole task horizon=5 (i/o 200), MPC planning horizon=5 (i/o 25)
elif [ "$1" = "cartpole_fast" ]; then
    python scripts/mbexp.py -env cartpole -ca model-type P -ca prop-type DS -o exp_cfg.exp_cfg.ntrain_iters 2 -o exp_cfg.sim_cfg.task_hor 5 -o ctrl_cfg.opt_cfg.plan_hor 5
# Fast run safety task horizon=5 (i/o 200), MPC planning horizon=5 (i/o 25)
elif [ "$1" = "safety_fast" ]; then
    python scripts/mbexp.py -env safety -ca model-type P -ca prop-type DS -o exp_cfg.exp_cfg.ntrain_iters 2 -o exp_cfg.sim_cfg.task_hor 5 -o ctrl_cfg.opt_cfg.plan_hor 5
# Run PPO, TRPO, and CPO on safety env
elif [ "$1" = "safety_agents" ]; then
    python experiment.py --algo 'trpo'
    python experiment.py --algo 'ppo'
    python experiment.py --algo 'cpo'
fi


#Notes
# planning horizon vs. task horizon?
# planning horizon is how far in advance MPC plans for
# task horizon is how many to mess around in the environment for

# model-type:
# D deterministic networks
# P probabilistic networks
# DE Deterministic ensemble
# PE Probabilistic ensemble (default)

# prop-type:
# E: deterministic
# DS: distribution sampling
# TS1: Trajectory sampling 1
# TSinf: Trajectory sampling infinite
# MM: Moment matching

#allowed combos:
# D-E : deterministic network and propagation
# P-E, P_DS, P-MM: Prob.All expectation, dist sampling, and moment matching
# DE-*: Can do all propagation types
# PE-*: Can do all propagation types

#Default settings
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