# Experiments
# Baseline mode #1: prob model, dist sampling propagation, and only 2 training iterations
python scripts/mbexp.py -env cartpole -ca model-type P -ca prop-type DS -o exp_cfg.exp_cfg.ntrain_iters 2

# #1 random shooting instead of CEM --> doesn't make it that much faster
python scripts/mbexp.py -env cartpole -ca model-type P -ca prop-type DS -ca opt-type Random -o exp_cfg.exp_cfg.ntrain_iters 2

# #1 + task horizon is only 5 (i/o 200), so you dont move around in the environment much.
python scripts/mbexp.py -env cartpole -ca model-type P -ca prop-type DS -o exp_cfg.exp_cfg.ntrain_iters 2 -o exp_cfg.sim_cfg.task_hor 5

# Fastest run!!!! Use this one to run really quick
# #1 + task horizon is only 5, MPC planning horizon is only 5 (i/o 25)
python scripts/mbexp.py -env cartpole -ca model-type P -ca prop-type DS -o exp_cfg.exp_cfg.ntrain_iters 2 -o exp_cfg.sim_cfg.task_hor 5 -o ctrl_cfg.opt_cfg.plan_hor 5

# Running with safety point goal 1
python scripts/mbexp.py -env safety_point_goal_1 -ca model-type P -ca prop-type DS -o exp_cfg.exp_cfg.ntrain_iters 2 -o exp_cfg.sim_cfg.task_hor 5 -o ctrl_cfg.opt_cfg.plan_hor 5

python scripts/mbexp.py -env safety_point_goal_1 -ca model-type P -ca prop-type DS -o exp_cfg.exp_cfg.ntrain_iters 200 -o exp_cfg.sim_cfg.task_hor 5 -o ctrl_cfg.opt_cfg.plan_hor 5

python scripts/mbexp.py -env safety_point_goal_1 -o exp_cfg.exp_cfg.ntrain_iters 200 -o exp_cfg.sim_cfg.task_hor 5 -o ctrl_cfg.opt_cfg.plan_hor 5

# Run vanilla PE-TS on safety point goal 1
python scripts/mbexp.py -env safety_point_goal_1

# Pretrain the neural network for 10 * 1000 times
python scripts/mbexp.py -env safety_point_goal_1 -o exp_cfg.exp_cfg.ninit_rollouts 10

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
{'ctrl_cfg': {'env': <dmbrl.env.cartpole.CartpoleEnv object at 0x7fae5fe8d470>,
              'opt_cfg': {'ac_cost_fn': <function CartpoleConfigModule.ac_cost_fn at 0x7fae6010b840>,
                          'cfg': {'alpha': 0.1,
                                  'max_iters': 5,
                                  'num_elites': 40,
                                  'popsize': 400},
                          'mode': 'CEM',
                          'obs_cost_fn': <function CartpoleConfigModule.obs_cost_fn at 0x7fae6010b7b8>,
                          'plan_hor': 25},
              'prop_cfg': {'mode': 'TSinf',
                           'model_init_cfg': {'model_class': <class 'dmbrl.modeling.models.BNN.BNN'>,
                                              'model_constructor': <bound method CartpoleConfigModule.nn_constructor of <cartpole.CartpoleConfigModule object at 0x7fae5fe8d0b8>>,
                                              'num_nets': 5},
                           'model_train_cfg': {'epochs': 5},
                           'npart': 20,
                           'obs_postproc': <function CartpoleConfigModule.obs_postproc at 0x7fae6010b6a8>,
                           'obs_preproc': <function CartpoleConfigModule.obs_preproc at 0x7fae6010b620>,
                           'targ_proc': <function CartpoleConfigModule.targ_proc at 0x7fae6010b730>}},
 'exp_cfg': {'exp_cfg': {'nrollouts_per_iter': 1, 'ntrain_iters': 50},
             'log_cfg': {'logdir': 'log'},
             'sim_cfg': {'env': <dmbrl.env.cartpole.CartpoleEnv object at 0x7fae5fe8d470>,
                         'task_hor': 200}}}