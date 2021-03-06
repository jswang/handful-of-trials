from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import argparse
import pprint

from dotmap import DotMap

from dmbrl.misc.MBExp import MBExperiment
from dmbrl.controllers.MPC import MPC
from dmbrl.config import create_config


def main(env, ctrl_type, ctrl_args, overrides, logdir):
    ctrl_args = DotMap(**{key: val for (key, val) in ctrl_args})
    cfg = create_config(env, ctrl_type, ctrl_args, overrides, logdir)
    cfg.pprint()

    if ctrl_type == "MPC":
        cfg.exp_cfg.exp_cfg.policy = MPC(cfg.ctrl_cfg)
    exp = MBExperiment(cfg.exp_cfg)

    os.makedirs(exp.logdir)
    with open(os.path.join(exp.logdir, "config.txt"), "w") as f:
        f.write(pprint.pformat(cfg.toDict()))

    exp.run_experiment()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-env', type=str, required=True,
                        help='Environment name: select from [cartpole, reacher, pusher, halfcheetah, safety_point_goal_1]')
    parser.add_argument('-ca', '--ctrl_arg', action='append', nargs=2, default=[],
                        help='Controller arguments, see https://github.com/kchua/handful-of-trials#controller-arguments')
    parser.add_argument('-o', '--override', action='append', nargs=2, default=[],
                        help='Override default parameters, see https://github.com/kchua/handful-of-trials#overrides')
    parser.add_argument('-logdir', type=str, default='data',
                        help='Directory to which results will be logged (default: ./data)')
    args = parser.parse_args()

    main(args.env, "MPC", args.ctrl_arg, args.override, args.logdir)
    # print("calling main")
    # print(args.ctrl_arg)
    # print(args.override)
    # print(args.logdir)

# jsw: Manual runs
# print("WARNING WARNING UNCOMMENT MBEXP.PY")
# main('safety_point_goal_1', "MPC", [], [],  './data/tmp')
# main('cartpole', "MPC", [], [], './data/tmp')
# main('safety_point_goal_1', "MPC",[['model-type', 'P'], ['prop-type', 'DS']] , [['exp_cfg.exp_cfg.ntrain_iters', '2'], ['exp_cfg.sim_cfg.task_hor', '5'], ['ctrl_cfg.opt_cfg.plan_hor', '5']]
# , './data')
