import sys
import csv
import lasagne

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
from sandbox.cpo.algos.safe.PDO_DDPG import PDO_DDPG
from sandbox.cpo.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from sandbox.cpo.safety_constraints.gather import GatherSafetyConstraint
from rllab.exploration_strategies.ou_strategy import OUStrategy
from rllab.q_functions.continuous_mlp_q_function import ContinuousMLPQFunction


ec2_mode = False


def run_task(*_):

        f = open('/home/qingkai/ddpg_performance.csv', "w+")

        env = PointGatherEnv(apple_reward=10,bomb_cost=1,n_apples=2, activity_range=6)

        policy = DeterministicMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=(64, 32)
        )

        es = OUStrategy(env_spec=env.spec)

        qf = ContinuousMLPQFunction(env_spec=env.spec)
        qf_cost = ContinuousMLPQFunction(env_spec=env.spec)


        safety_constraint = GatherSafetyConstraint(max_value=0.1)



        algo = PDO_DDPG(
            env=env,
            policy=policy,
            es=es,
            qf=qf,
            qf_cost=qf_cost,
            dual_var=0,
            safety_constraint=safety_constraint,
            batch_size=128,
            max_path_length=15,
            epoch_length=10000,
            min_pool_size=10000,
            n_epochs=200,
            discount=0.99,
            qf_learning_rate=1e-3,
            qf_cost_learning_rate=1e-3,
            dual_learning_rate=1e-2,
            policy_learning_rate=1e-3,
            scale_reward=1,
            scale_cost=10,
            soft_target=True,
            soft_target_tau=0.001,
            eval_samples=10000,
            #plot=True,
        )

        algo.train()
        f.close()


run_experiment_lite(
    run_task,
    n_parallel=4,
    snapshot_mode="last",
    exp_prefix='PDO_DDPG-PointGather',
    seed=1,
    mode = "ec2" if ec2_mode else "local"
    #plot=True
)
