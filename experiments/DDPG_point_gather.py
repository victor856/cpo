import sys
import csv

sys.path.append(".")

from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
import lasagne.nonlinearities as NL

# Policy
#from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.policies.deterministic_mlp_policy import DeterministicMLPPolicy

# Baseline
#from sandbox.cpo.baselines.gaussian_mlp_baseline import GaussianMLPBaseline

# Environment
from sandbox.cpo.envs.mujoco.gather.point_gather_env import PointGatherEnv

# Policy optimization
from sandbox.cpo.algos.safe.pdo_ddpg import PDO_DDPG
from sandbox.cpo.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from sandbox.cpo.safety_constraints.gather import GatherSafetyConstraint
from rllab.exploration_strategies.ou_strategy import OUStrategy
from rllab.q_functions.continuous_mlp_q_function import ContinuousMLPQFunction


ec2_mode = False


def run_task(*_):
        #trpo_stepsize = 0.01
        #trpo_subsample_factor = 0.2
        
        env = PointGatherEnv(apple_reward=10,bomb_cost=1,n_apples=2, activity_range=6)

        #policy = GaussianMLPPolicy(env.spec,
                    #hidden_sizes=(64,32)
                    #)
        policy = DeterministicMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=(64, 32)
        )

        es = OUStrategy(env_spec=env.spec)

        qf = ContinuousMLPQFunction(env_spec=env.spec)
        qf_cost = ContinuousMLPQFunction(env_spec=env.spec)

        #baseline = GaussianMLPBaseline(
            #env_spec=env.spec,
            #regressor_args={
                    #'hidden_sizes': (64,32),
                    #'hidden_nonlinearity': NL.tanh,
                    #'learn_std':False,
                    #'step_size':trpo_stepsize,
                    #'optimizer':ConjugateGradientOptimizer(subsample_factor=trpo_subsample_factor)
                    #}
        #)

        safety_constraint = GatherSafetyConstraint(max_value=0.1)



        algo = PDO_DDPG(
            env=env,
            policy=policy,
            es=es,
            qf=qf,
            qf_cost=qf_cost,
            dual_var=0,
            #baseline=baseline,
            safety_constraint=safety_constraint,
            batch_size=32,
            max_path_length=100,
            epoch_length=1000,
            min_pool_size=10000,
            n_epochs=200,
            #n_itr=100,
            #gae_lambda=0.95,
            discount=0.99,
            qf_weight_decay=0.,
            qf_update_method='adam',
            qf_cost_update_method='adam',
            qf_learning_rate=1e-3,
            qf_cost_learning_rate=1e-3,
            dual_learning_rate=0.01,
            policy_weight_decay=0,
            policy_update_method='adam',
            policy_learning_rate=1e-4,
            scale_reward=1.0,
            scale_cost=1.0,
            soft_target=True,
            soft_target_tau=0.001,
            #plot=True,
        )

        algo.train()


run_experiment_lite(
    run_task,
    n_parallel=2,
    snapshot_mode="last",
    exp_prefix='PDO_DDPG-PointGather',
    seed=1,
    mode = "ec2" if ec2_mode else "local"
    #plot=True
)
