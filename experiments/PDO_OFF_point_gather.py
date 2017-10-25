import sys
import csv

sys.path.append(".")

from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
import lasagne.nonlinearities as NL

# Policy
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.policies.deterministic_mlp_policy import DeterministicMLPPolicy

# Baseline
from sandbox.cpo.baselines.gaussian_mlp_baseline import GaussianMLPBaseline

# Environment
from sandbox.cpo.envs.mujoco.gather.point_gather_env import PointGatherEnv

# Policy optimization
from sandbox.cpo.algos.safe.pdo_offpolicy import PDO_OFF
from sandbox.cpo.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from sandbox.cpo.safety_constraints.gather import GatherSafetyConstraint

from rllab.exploration_strategies.ou_strategy import OUStrategy
from rllab.q_functions.continuous_mlp_q_function import ContinuousMLPQFunction




ec2_mode = False


def run_task(*_):
        f = open('/home/qingkai/verina.csv', "w+")
        trpo_stepsize = 0.01
        trpo_subsample_factor = 0.2
        
        env = PointGatherEnv(apple_reward=10,bomb_cost=1,n_apples=2, activity_range=6)

        policy = GaussianMLPPolicy(env.spec,
                    hidden_sizes=(64,32)
                 )

        baseline = GaussianMLPBaseline(
            env_spec=env.spec,
            regressor_args={
                    'hidden_sizes': (64,32),
                    'hidden_nonlinearity': NL.tanh,
                    'learn_std':False,
                    'step_size':trpo_stepsize,
                    'optimizer':ConjugateGradientOptimizer(subsample_factor=trpo_subsample_factor)
                    }
        )

        safety_constraint = GatherSafetyConstraint(max_value=0.2)
        
        ddpg_policy = DeterministicMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=(64, 32)
        )

        ddpg_es = OUStrategy(env_spec=env.spec)

        ddpg_qf = ContinuousMLPQFunction(env_spec=env.spec)
        ddpg_qf_cost = ContinuousMLPQFunction(env_spec=env.spec)


        algo = PDO_OFF(
            env=env,
            policy=policy,
            baseline=baseline,
            safety_constraint=safety_constraint,
            batch_size=10000,
            max_path_length=15,
            n_itr=200,
            gae_lambda=0.95,
            discount=0.995,
            step_size=trpo_stepsize,
            optimizer_args={'subsample_factor':trpo_subsample_factor},
            ddpg_policy=ddpg_policy,
            ddpg_qf=ddpg_qf,
            ddpg_qf_cost=ddpg_qf_cost,            
            ddpg_es=ddpg_es,
            ddpg_dual_var=0,
            ddpg_batch_size=64,
            ddpg_qf_learning_rate=1e-3,
            ddpg_qf_cost_learning_rate=1e-3,
            ddpg_dual_learning_rate=1e-3,
            ddpg_policy_learning_rate=1e-3,
            ddpg_scale_reward=1,
            ddpg_scale_cost=10,
            offline_itr_n=50000,
            balance=0,
            safety_tradeoff_coeff_lr=1e-2,    
            avg_horizon=200,
            adjust_epoch=5,      
            #plot=True,
        )

        algo.train()
        f.close()


run_experiment_lite(
    run_task,
    n_parallel=4,
    snapshot_mode="last",
    exp_prefix='PDO-OFF-PointGather',
    seed=1,
    mode = "ec2" if ec2_mode else "local"
    #plot=True
)

