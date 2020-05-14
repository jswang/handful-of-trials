import gym
import safety_gym

class SafetyPointGoal1ConfigModule:
    ENV_NAME = 'Safexp-PointGoal1-v0'
    # copied from cartpole
    TASK_HORIZON = 200
    NTRAIN_ITERS = 50
    NROLLOUTS_PER_ITER = 1
    PLAN_HOR = 25
    # 60 state points, 2 action points
    MODEL_IN, MODEL_OUT = 62, 60
    GP_NINDUCING_POINTS = 200

    def __init__(self):
        self.ENV = gym.make(self.ENV_NAME)

        # copied from cartpole
        cfg = tf.ConfigProto()
        cfg.gpu_options.allow_growth = True
        self.SESS = tf.Session(config=cfg)
        self.NN_TRAIN_CFG = {"epochs": 5}
        self.OPT_CFG = {
            "Random": {
                "popsize": 2000
            },
            "CEM": {
                "popsize": 400,
                "num_elites": 40,
                "max_iters": 5,
                "alpha": 0.1
            }
        }

    # These appear to be additional functions needed by MPC and GP
    # copied from half cheetah

    @staticmethod
    # this seems to be taking the observations from the env and taking the sin and cos of them for MPC
    # not entirely sure, and it's optional, so ignore for now.
    # def obs_preproc(obs):
    #     if isinstance(obs, np.ndarray):
    #         return np.concatenate([obs[:, 1:2], np.sin(obs[:, 2:3]), np.cos(obs[:, 2:3]), obs[:, 3:]], axis=1)
    #     else:
    #         return tf.concat([obs[:, 1:2], tf.sin(obs[:, 2:3]), tf.cos(obs[:, 2:3]), obs[:, 3:]], axis=1)

    # @staticmethod
    # def obs_postproc(obs, pred):
    #     if isinstance(obs, np.ndarray):
    #         return np.concatenate([pred[:, :1], obs[:, 1:] + pred[:, 1:]], axis=1)
    #     else:
    #         return tf.concat([pred[:, :1], obs[:, 1:] + pred[:, 1:]], axis=1)

    # .targ_proc (func): (optional) A function which takes current observations and next
    # observations and returns the array of targets (so that the model learns the mapping
    # obs -> targ_proc(obs, next_obs)). Defaults to lambda obs, next_obs: next_obs.
    # Note: Only needs to process NumPy arrays.
    # @staticmethod
    # def targ_proc(obs, next_obs):
    #     return np.concatenate([next_obs[:, :1], next_obs[:, 1:] - obs[:, 1:]], axis=1)

    @staticmethod
    def obs_cost_fn(obs):
        return -obs[:, 0]

    @staticmethod
    def ac_cost_fn(acs):
        if isinstance(acs, np.ndarray):
            return 0.1 * np.sum(np.square(acs), axis=1)
        else:
            return 0.1 * tf.reduce_sum(tf.square(acs), axis=1)

    def nn_constructor(self, model_init_cfg):
        model = get_required_argument(model_init_cfg, "model_class", "Must provide model class")(DotMap(
            name="model", num_networks=get_required_argument(model_init_cfg, "num_nets", "Must provide ensemble size"),
            sess=self.SESS, load_model=model_init_cfg.get("load_model", False),
            model_dir=model_init_cfg.get("model_dir", None)
        ))
        if not model_init_cfg.get("load_model", False):
            model.add(FC(200, input_dim=self.MODEL_IN, activation="swish", weight_decay=0.000025))
            model.add(FC(200, activation="swish", weight_decay=0.00005))
            model.add(FC(200, activation="swish", weight_decay=0.000075))
            model.add(FC(200, activation="swish", weight_decay=0.000075))
            model.add(FC(self.MODEL_OUT, weight_decay=0.0001))
        model.finalize(tf.train.AdamOptimizer, {"learning_rate": 0.001})
        return model

    def gp_constructor(self, model_init_cfg):
        model = get_required_argument(model_init_cfg, "model_class", "Must provide model class")(DotMap(
            name="model",
            kernel_class=get_required_argument(model_init_cfg, "kernel_class", "Must provide kernel class"),
            kernel_args=model_init_cfg.get("kernel_args", {}),
            num_inducing_points=get_required_argument(
                model_init_cfg, "num_inducing_points", "Must provide number of inducing points."
            ),
            sess=self.SESS
        ))
        return model


CONFIG_MODULE = SafetyPointGoal1ConfigModule