import sys

sys.path.append(".")

from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
import lasagne.nonlinearities as NL

# Policy
#from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.policies.deterministic_mlp_policy import DeterministicMLPPolicy

# Baseline
from sandbox.cpo.baselines.gaussian_mlp_baseline import GaussianMLPBaseline

# Environment
from sandbox.cpo.envs.mujoco.gather.point_gather_env import PointGatherEnv

# Policy optimization
from sandbox.cpo.algos.safe.pdo_ddpg import PDODDPG
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

        #safety_constraint = GatherSafetyConstraint(max_value=0.1)



        algo = DDPG(
            env=env,
            policy=policy,
            es=es,
            qf=qf,
            #baseline=baseline,
            #safety_constraint=safety_constraint,
            batch_size=32,
            max_path_length=100,
            epoch_length=1000,
            min_pool_size=10000,
            n_epochs=200,
            #n_itr=100,
            #gae_lambda=0.95,
            discount=0.99,
            qf_learning_rate=1e-3,
            policy_learning_rate=1e-4,
            scale_reward=0.01,
            soft_target=True,
            soft_target_tau=0.001,
            step_size=trpo_stepsize,
            optimizer_args={'subsample_factor':trpo_subsample_factor},
            #plot=True,
        )

        algo.train()


run_experiment_lite(
    run_task,
    n_parallel=1,
    snapshot_mode="last",
    exp_prefix='DDPG-PointGather',
    seed=1,
    mode = "ec2" if ec2_mode else "local"
    #plot=True
)
