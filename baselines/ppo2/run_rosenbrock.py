#!/usr/bin/env python3
import sys
from baselines import bench, logger
from baselines.common.cmd_util import make_rosenbrock_env, rosi_arg_parser
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.ppo2 import ppo2
from baselines.ppo2.policies import CnnPolicy, LstmPolicy, LnLstmPolicy, MlpPolicy
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common import set_global_seeds
import multiprocessing
import tensorflow as tf
import gym
from gym.envs.registration import EnvSpec


def train(env_id, num_timesteps, seed):
    from baselines.common import set_global_seeds
    from baselines.common.vec_env.vec_normalize import VecNormalize
    from baselines.ppo2 import ppo2
    from baselines.ppo2.policies import MlpPolicy
    import gym
    import tensorflow as tf
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
    ncpu = 2
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf.Session(config=config).__enter__()

    def make_env():
        env = gym.make(env_id)
        env = bench.Monitor(env, logger.get_dir(), allow_early_resets=True)
        return env

    env = DummyVecEnv([make_env])
    env = VecNormalize(env)

    set_global_seeds(seed)
    policy = MlpPolicy
    model = ppo2.learn(policy=policy, env=env, nsteps=512, nminibatches=32,
                       lam=0.95, gamma=0.99, noptepochs=10, log_interval=1,
                       ent_coef=0.0,
                       lr=3e-4,
                       cliprange=0.2,
                       total_timesteps=num_timesteps)

    return model, env


def test(env_id, model_path):
    return done, state

def main():
    flag = "train"
    parser = rosi_arg_parser()
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm', 'mlp'], default='lstm')
    #parser.add_argument('--env', help='environment ID', default='Rosi-v0')
    args = parser.parse_args()
    logger.configure()
    if flag=="train":
        model, env = train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)
    else:
        model_path = 0
        done, state = test(args.env, model_path)

if __name__ == '__main__':
    main()
