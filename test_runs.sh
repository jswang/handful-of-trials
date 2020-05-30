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

# Fast run with safety point goal
python scripts/mbexp.py -env safety_point_goal_1 -ca model-type P -ca prop-type DS -o exp_cfg.exp_cfg.ntrain_iters 2 -o exp_cfg.sim_cfg.task_hor 5 -o ctrl_cfg.opt_cfg.plan_hor 5

python scripts/mbexp.py -env safety_point_goal_1 -ca model-type P -ca prop-type DS -o exp_cfg.exp_cfg.ntrain_iters 200 -o exp_cfg.sim_cfg.task_hor 5 -o ctrl_cfg.opt_cfg.plan_hor 5

python scripts/mbexp.py -env safety_point_goal_1 -o exp_cfg.exp_cfg.ntrain_iters 200 -o exp_cfg.sim_cfg.task_hor 5 -o ctrl_cfg.opt_cfg.plan_hor 5

# Run vanilla PE-TS on safety point goal 1
python scripts/mbexp.py -env safety_point_goal_1

# Pretrain the neural network for 10 * 1000 times
python scripts/mbexp.py -env safety_point_goal_1 -o exp_cfg.exp_cfg.ninit_rollouts 10

# Gather 5 *1000 timesteps, then train for 10 epochs
python scripts/mbexp.py -env safety_point_goal_1 -o exp_cfg.exp_cfg.ninit_rollouts 5 -o ctrl_cfg.prop_cfg.model_train_cfg.epochs 10 -o ctrl_cfg.prop_cfg.model_train_cfg

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

# Cartpole
# python scripts/mbexp.py -env cartpole -o exp_cfg.exp_cfg.ninit_rollouts 5 -o ctrl_cfg.prop_cfg.model_train_cfg.epochs 10
# Network training:  10%| | 1/10 [00:00<00:05,  1.71epoch(s)/s, Training loss(es)=[0.1808334  0.2879471  0.21516424 0.19267736 0.2314286
# Network training:  10%| | 1/10 [00:00<00:08,  1.02epoch(s)/s, Training loss(es)=[0.17715067 0.25897726 0.1648757  0.1779611  0.2139903
# Network training:  20%|▏| 2/10 [00:00<00:03,  2.03epoch(s)/s, Training loss(es)=[0.17715067 0.25897726 0.1648757  0.1779611  0.2139903
# Network training:  20%|▏| 2/10 [00:01<00:05,  1.43epoch(s)/s, Training loss(es)=[0.16049469 0.23022838 0.15109587 0.134102   0.2020068
# Network training:  30%|▎| 3/10 [00:01<00:03,  2.15epoch(s)/s, Training loss(es)=[0.16049469 0.23022838 0.15109587 0.134102   0.2020068
# Network training:  30%|▎| 3/10 [00:01<00:04,  1.66epoch(s)/s, Training loss(es)=[0.27331612 0.23652203 0.14023213 0.10356279 0.1906842
# Network training:  40%|▍| 4/10 [00:01<00:02,  2.21epoch(s)/s, Training loss(es)=[0.27331612 0.23652203 0.14023213 0.10356279 0.1906842
# Network training:  40%|▍| 4/10 [00:02<00:03,  1.80epoch(s)/s, Training loss(es)=[0.13444144 0.20210059 0.17017294 0.1065122  0.1661615
# Network training:  50%|▌| 5/10 [00:02<00:02,  2.25epoch(s)/s, Training loss(es)=[0.13444144 0.20210059 0.17017294 0.1065122  0.1661615
# Network training:  50%|▌| 5/10 [00:02<00:02,  1.90epoch(s)/s, Training loss(es)=[0.10226434 0.21457402 0.12141062 0.10204402 0.1649454
# Network training:  60%|▌| 6/10 [00:02<00:01,  2.28epoch(s)/s, Training loss(es)=[0.10226434 0.21457402 0.12141062 0.10204402 0.1649454
# Network training:  60%|▌| 6/10 [00:03<00:02,  1.97epoch(s)/s, Training loss(es)=[0.09203486 0.18885158 0.11010803 0.1002076  0.1628701
# Network training:  70%|▋| 7/10 [00:03<00:01,  2.30epoch(s)/s, Training loss(es)=[0.09203486 0.18885158 0.11010803 0.1002076  0.1628701
# Network training:  70%|▋| 7/10 [00:03<00:01,  2.03epoch(s)/s, Training loss(es)=[0.08904738 0.18573822 0.12231557 0.10402434 0.1599710
# Network training:  80%|▊| 8/10 [00:03<00:00,  2.32epoch(s)/s, Training loss(es)=[0.08904738 0.18573822 0.12231557 0.10402434 0.1599710
# Network training:  80%|▊| 8/10 [00:03<00:00,  2.06epoch(s)/s, Training loss(es)=[0.0899111  0.16203672 0.11181843 0.10373364 0.1420228
# Network training:  90%|▉| 9/10 [00:03<00:00,  2.31epoch(s)/s, Training loss(es)=[0.0899111  0.16203672 0.11181843 0.10373364 0.1420228
# Network training:  90%|▉| 9/10 [00:04<00:00,  2.09epoch(s)/s, Training loss(es)=[0.11653279 0.1653193  0.11957336 0.10071964 0.1487318
# Network training: 100%|█| 10/10 [00:04<00:00,  2.32epoch(s)/s, Training loss(es)=[0.11653279 0.1653193  0.11957336 0.10071964 0.14873187]]

# Half cheetah
# python scripts/mbexp.py -env halfcheetah -o exp_cfg.exp_cfg.ninit_rollouts 5 -o ctrl_cfg.prop_cfg.model_train_cfg.epochs 10
# Network training:   0%| | 0/10 [00:00<?, ?epoch(s)/s, Training loss(es)=[28.699594 27.28497  27.635157 28
# Network training:  10%| | 1/10 [00:00<00:07,  1.20epoch(s)/s, Training loss(es)=[28.699594 27.28497  27.6
# Network training:  10%| | 1/10 [00:01<00:14,  1.57s/epoch(s), Training loss(es)=[10.913377   9.56528   11
# Network training:  20%|▏| 2/10 [00:01<00:06,  1.27epoch(s)/s, Training loss(es)=[10.913377   9.56528   11.705
# Network training:  20%|▏| 2/10 [00:02<00:09,  1.17s/epoch(s), Training loss(es)=[3.7992117 4.719084  6.652
# Network training:  30%|▎| 3/10 [00:02<00:05,  1.29epoch(s)/s, Training loss(es)=[3.7992117 4.719084  6.652
# Network training:  30%|▎| 3/10 [00:03<00:07,  1.03s/epoch(s), Training loss(es)=[2.483982  2.392428  2.557
# Network training:  40%|▍| 4/10 [00:03<00:04,  1.30epoch(s)/s, Training loss(es)=[2.483982  2.392428  2.557
# Network training:  40%|▍| 4/10 [00:03<00:05,  1.05epoch(s)/s, Training loss(es)=[1.7740941 1.5752999 2.261
# Network training:  50%|▌| 5/10 [00:03<00:03,  1.31epoch(s)/s, Training loss(es)=[1.7740941 1.5752999 2.261
# Network training:  50%|▌| 5/10 [00:04<00:04,  1.10epoch(s)/s, Training loss(es)=[1.1480645 1.0897961 1.822
# Network training:  60%|▌| 6/10 [00:04<00:03,  1.31epoch(s)/s, Training loss(es)=[1.1480645 1.0897961 1.822
# Network training:  60%|▌| 6/10 [00:05<00:03,  1.13epoch(s)/s, Training loss(es)=[1.0462909  0.99667966 1.2
# Network training:  70%|▋| 7/10 [00:05<00:02,  1.32epoch(s)/s, Training loss(es)=[1.0462909  0.99667966 1.2
# Network training:  70%|▋| 7/10 [00:06<00:02,  1.15epoch(s)/s, Training loss(es)=[0.9617508 0.8996628 1.032
# Network training:  80%|▊| 8/10 [00:06<00:01,  1.32epoch(s)/s, Training loss(es)=[0.9617508 0.8996628 1.032
# Network training:  80%|▊| 8/10 [00:06<00:01,  1.18epoch(s)/s, Training loss(es)=[0.9203289  0.8484211  0.9
# Network training:  90%|▉| 9/10 [00:06<00:00,  1.32epoch(s)/s, Training loss(es)=[0.9203289  0.8484211  0.9
# Network training:  90%|▉| 9/10 [00:07<00:00,  1.19epoch(s)/s, Training loss(es)=[0.8893338  0.8373271  0.8
# Network training: 100%|█| 10/10 [00:07<00:00,  1.33epoch(s)/s, Training loss(es)=[0.8893338  0.8373271  0.8737886  0.8644587  0.83382267]]


# Pusher
# Network training:  10%| | 1/10 [00:00<00:02,  4.41epoch(s)/s, Training loss(es)=[0.20496745 0.20271729 0.20732684 0.22522634 0.2225378
# Network training:  10%| | 1/10 [00:00<00:02,  3.20epoch(s)/s, Training loss(es)=[0.15656753 0.1713661  0.15314569 0.18398969 0.1742803
# Network training:  10%| | 1/10 [00:00<00:03,  2.58epoch(s)/s, Training loss(es)=[0.14114292 0.15881048 0.13987572 0.16955264 0.1553550
# Network training:  30%|▎| 3/10 [00:00<00:00,  7.74epoch(s)/s, Training loss(es)=[0.14114292 0.15881048 0.13987572 0.16955264 0.1553550
# Network training:  30%|▎| 3/10 [00:00<00:01,  6.49epoch(s)/s, Training loss(es)=[0.12890695 0.14993502 0.12348779 0.15190004 0.1376004
# Network training:  30%|▎| 3/10 [00:00<00:01,  5.61epoch(s)/s, Training loss(es)=[0.11699543 0.14442003 0.11459671 0.14254202 0.1126084
# Network training:  50%|▌| 5/10 [00:00<00:00,  9.35epoch(s)/s, Training loss(es)=[0.11699543 0.14442003 0.11459671 0.14254202 0.1126084
# Network training:  50%|▌| 5/10 [00:00<00:00,  8.20epoch(s)/s, Training loss(es)=[0.10385827 0.13376069 0.10090273 0.13101752 0.0964496
# Network training:  50%|▌| 5/10 [00:00<00:00,  7.30epoch(s)/s, Training loss(es)=[0.09521569 0.12149235 0.09150259 0.12227293 0.0875146
# Network training:  70%|▋| 7/10 [00:00<00:00, 10.22epoch(s)/s, Training loss(es)=[0.09521569 0.12149235 0.09150259 0.12227293 0.0875146
# Network training:  70%|▋| 7/10 [00:00<00:00,  9.22epoch(s)/s, Training loss(es)=[0.09019681 0.11782528 0.0849393  0.11310694 0.0840979
# Network training:  70%|▋| 7/10 [00:00<00:00,  8.39epoch(s)/s, Training loss(es)=[0.08450385 0.11355101 0.07804384 0.10476902 0.0787423
# Network training:  90%|▉| 9/10 [00:00<00:00, 10.79epoch(s)/s, Training loss(es)=[0.08450385 0.11355101 0.07804384 0.10476902 0.0787423
# Network training:  90%|▉| 9/10 [00:00<00:00,  9.90epoch(s)/s, Training loss(es)=[0.07667151 0.10854874 0.07613518 0.09699158 0.0757219
# Network training: 100%|█| 10/10 [00:00<00:00, 11.00epoch(s)/s, Training loss(es)=[0.07667151 0.10854874 0.07613518 0.09699158 0.07572193]]

# Reacher
# python scripts/mbexp.py -env reacher -o exp_cfg.exp_cfg.ninit_rollouts 5 -o ctrl_cfg.prop_cfg.model_train_cfg.epochs 10
# training loss starts at 5, and goes down to like .20, then down all the way to like .13 slowly