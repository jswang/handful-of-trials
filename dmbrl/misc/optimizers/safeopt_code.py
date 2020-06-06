from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import numpy as np
import scipy.stats as stats
import GPy
from .optimizer import Optimizer
import tensorflow.contrib.eager as tfe
from dmbrl.misc.optimizers.SafeOpt.safeopt.gp_opt import SafeOptSwarm

class SafeOptimizer(Optimizer):
    """A Tensorflow-compatible GP based safe optimizer.
    """
    def __init__(self, sol_dim, max_iters, swarmsize=30, tf_session=None,
                 upper_bound=None,lower_bound=None, epsilon=0.001, beta=2):
        """Creates an instance of this class.

        Arguments:
            sol_dim (int): The dimensionality of the problem space
            max_iters (int): The maximum number of iterations to perform during optimization
            SwarmSize (int): Swarm size of approximate GP optimizer
            tf_session (tf.Session): (optional) Session to be used for this optimizer. Defaults to None,
                in which case any functions passed in cannot be tf.Tensor-valued.
            bounds (2x1 np.array): bounds for action space
            epsilon (float): A minimum variance. If the maximum variance drops below epsilon, optimization is
                stopped.
            beta (float): Controls size of confidence interval
        """
        super().__init__()
        self.sol_dim, self.max_iters, self.swarm_size = sol_dim, max_iters, swarmsize
        bounds=[]
        for i in range(len(upper_bound)):
            bounds.append([lower_bound[i],upper_bound[i]])
        self.bounds = bounds
        self.epsilon, self.beta = epsilon, beta
        self.tf_sess = tf_session
        self.fmin=-10**6 # change later
        if self.tf_sess is not None:
            with self.tf_sess.graph.as_default():
                with tf.variable_scope("SafeOptSolver") as scope:
                    self.init_pt = tf.placeholder(dtype=tf.float32, shape=[sol_dim])
                    self.init_var = tf.placeholder(dtype=tf.float32, shape=[sol_dim])

        self.num_opt_iters, self.pt, self.var = None, None, None
        self.tf_compatible, self.cost_function = None, None
        #Initialization of Points- Custom
        init_pt=np.zeros((sol_dim,1)).T

        sample=np.array([[0]])
        #--------------------------
        self.gpmodel=GPy.models.GPRegression(init_pt,sample,noise_var=0.001)
        self.opt=SafeOptSwarm(self.gpmodel,self.fmin,self.bounds,swarm_size=self.swarm_size)
        #self.opt=SafeOpt(self.gpmodel,self.fmin,self.bounds)
    def setup(self, cost_function, tf_compatible):
        """Sets up this optimizer using a given cost function.

        Arguments:
            cost_function (func): A function for computing costs over a batch of candidate solutions.
            tf_compatible (bool): True if the cost function provided is tf.Tensor-valued.

        Returns: None
        """
        if tf_compatible and self.tf_sess is None:
            raise RuntimeError("Cannot pass in a tf.Tensor-valued cost function without passing in a TensorFlow "
                               "session into the constructor")

        self.tf_compatible = tf_compatible

        if not tf_compatible:
            self.cost_function = cost_function
        else:
            def continue_optimization(t, pt, var, best_val, best_sol):
                return tf.logical_and(tf.less(t, self.max_iters), tf.reduce_max(var) > self.epsilon)

            def iteration(t, pt, var, best_val, best_sol):

                def step_safeopt(t,pt,cost,best_val,best_sol):
                    print("---------------Step of Safeopt-------------")

                    #Convert to np
                    pt_np=np.array(pt)
                    cost_np=np.array(cost)
                    best_val=np.array(best_val)
                    best_sol=np.array(best_sol)

                    print("Cost:{}".format(cost_np))
                    print("Best Cost: {}".format(best_val))
                    self.opt.add_new_data_point(pt_np,-cost_np)
                    new_pt,stddev=self.opt.get_new_query_point("expanders") # Returns point most likely to expand safe set

                    best_sol2,stddev2=self.opt.get_new_query_point("maximizers") # returns best parameters from current known points
                    #best_sol2=self.opt.optimize(ucb=True)


                    print("Expander Point Found: {}".format(new_pt))
                    print("Maximizer Point Point:{}".format(best_sol2))



                    print("-----------Step Ended------------------")
                    return new_pt,best_sol2
                pt=tf.expand_dims(pt,axis=0)
                cost=cost_function(pt)
                new_pt,best_sol2 = tfe.py_func(func=step_safeopt, inp=[t,pt,cost,best_val,best_sol], Tout=[pt.dtype,best_sol.dtype])
                best_sol2=tf.convert_to_tensor(best_sol2)
                best_sol2.set_shape(best_sol.get_shape())
                new_pt=tf.convert_to_tensor(new_pt)
                new_pt.set_shape(pt.get_shape())
                new_cost=cost_function(new_pt)
                new_pt=tf.squeeze(new_pt) # retain shape of pt
                new_best_cost=tf.squeeze(new_cost)
                return t + 1, new_pt, var, new_best_cost, best_sol2 # var has no significance

            with self.tf_sess.graph.as_default():
                '''
                @tf.function
                def wrapper_func(self):
                    @tf.function
                    def initialize_gp_local(init_pt,sample):
                        print("Entered initialize GP Local----------------")
                        init_pt=init_pt.eval()
                        sample=sample.eval()
                        print("Sample cost is -------------------------")
                        print(sample)
                        self.gpmodel=GPy.models.GPRegression(init_pt,sample,noise_var=0.001)
                        self.opt=SafeOptSwarm(gpmodel,self.fmin,self.bounds,swarm_size=self.swarm_size)
                        print("Succesfully initialized GP for Safeopt-----------------------------------")
                        return tf.float32(0)
                    print("Wrapper called")

                    return tf.py_function(initialize_gp_local,[init_pt,sample],tf.float32)
                    #return initialize_gp_local(init_pt,sample)
                print("Entered Session----------------------------")
                init_pt=tf.expand_dims(self.init_pt,axis=0) # Necessary to maintain dimensionality
                sample=-cost_function(init_pt) # cost for init_pt
                sample=tf.expand_dims(sample,axis=1)
                '''

                self.num_opt_iters, self.pt, self.var, self.best_val, self.best_sol = tf.while_loop(
                    cond=continue_optimization, body=iteration,
                    loop_vars=[0, self.init_pt, self.init_var, float("inf"), self.init_pt]
                )

    def reset(self):
        pass

    def obtain_solution(self, init_pt, init_var):



        if self.tf_compatible:
            #print(init_pt.get_shape())
            #print(init_var.get_shape())
            sol, solvar = self.tf_sess.run(
                [self.best_sol, self.var],
                feed_dict={self.init_pt: init_pt, self.init_var: init_var}
            )
        else:
            pt, var, t = init_pt, init_var, 0

            self.gpmodel=GPy.models.GPRegression(pt,self.cost_function(pt),noise_var=0.001)
            opt=SafeOptSwarm(self.gpmodel,fmin)

            while (t < self.max_iters) and np.max(var) > self.epsilon:

                new_pt,std_dev=opt.get_new_query_point("expanders") # Returns point most likely to expand safe set

                best_val,best_sol=opt.get_new_query_point("maximizer") # returns best parameters from current known points

                t += 1


            sol = best_sol
        return sol
