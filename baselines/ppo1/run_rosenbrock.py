#!/usr/bin/env python3
import os
from baselines.common.cmd_util import make_rosenbrock_env, rosi_arg_parser
from baselines.common import tf_util as U
from baselines import logger
from baselines.ppo1 import lstm_policy, pposgd_simple
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.ppo2.policies import LstmPolicy2
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
import tensorflow as tf
import time
import numpy as np

import gym

def policy_fn(name, ob_space, ac_space):
    return lstm_policy.LSTMPolicy(scope=name, reuse=False,
                                     ob_space=ob_space,
                                     ac_space=ac_space,
                                     hiddens=[128, 128], normalize=True)
    #return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
    #    hid_size=64, num_hid_layers=2)

def train(env_id, num_timesteps, seed, model_path=None):

    U.make_session(num_cpu=1).__enter__()
    def policy_fn(name, ob_space, ac_space):
        return lstm_policy.LSTMPolicy(scope=name, reuse=False,
                                         ob_space=ob_space,
                                         ac_space=ac_space,
                                         hiddens=[128, 128], normalize=True)
        #return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
        #    hid_size=64, num_hid_layers=2)
    env = make_rosenbrock_env(env_id, seed)

    # parameters below were the best found in a simple random search
    # these are good enough to make humanoid walk, but whether those are
    # an absolute best or not is not certain
    env = RewScale(env, 0.1)
    #env = DummyVecEnv([env])
    #env = VecNormalize(env)
    # timesteps_per_actorbatch=512,
    # optim_epochs=10,
    pi = pposgd_simple.learn(env, policy_fn,
            max_timesteps=num_timesteps,
            timesteps_per_actorbatch=2048,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10,
            optim_stepsize=3e-4,
            optim_batchsize=64,
            gamma=0.99,
            lam=0.95,
            schedule='linear',
        )
    env.close()
    model_path = "/Users/romc/Documents/RNN_exploation_learning/baselines/baselines/ppo1/tmp/test2"
    if model_path:
        U.save_state(model_path)

    return pi

class RewScale(gym.RewardWrapper):
    def __init__(self, env, scale):
        gym.RewardWrapper.__init__(self, env)
        self.scale = scale
    def reward(self, r):
        return r * self.scale


def test(env_id, model):

    # initialize env
    env = gym.make(env_id)
    ac = env.action_space.sample()  # not used, just so we have the datatype

    # initialize model
    model = U.load_state("/Users/romc/Documents/RNN_exploation_learning/baselines/baselines/ppo1/tmp/test2")

    #initial starting point
    state = np.array([2,2])


    done = False
    dist = 0
    count = 0
    while dist > 0.05 or count>100 or dist > 10000:

        # get action
        #action = model.step(self.obs, self.states, self.dones)
        action = 0

        # make a step
        state = env.step(action)
        print(state)

    if dist <= 0.05:
        done = True

    # Sample one trajectory (until trajectory end)
    def traj_1_generator(pi, env, horizon, stochastic):

        t = 0
        ac = env.action_space.sample()  # not used, just so we have the datatype
        new = True  # marks if we're on first timestep of an episode

        ob = env.reset()
        cur_ep_ret = 0  # return in current episode
        cur_ep_len = 0  # len of current episode

        # Initialize history arrays
        obs = []
        rews = []
        news = []
        acs = []

        while True:
            ac, vpred = pi.act(stochastic, ob)
            obs.append(ob)
            news.append(new)
            acs.append(ac)

            ob, rew, new, _ = env.step(ac)
            rews.append(rew)

            cur_ep_ret += rew
            cur_ep_len += 1
            if new or t >= horizon:
                break
            t += 1

        obs = np.array(obs)
        rews = np.array(rews)
        news = np.array(news)
        acs = np.array(acs)
        traj = {"ob": obs, "rew": rews, "new": news, "ac": acs,
                "ep_ret": cur_ep_ret, "ep_len": cur_ep_len}
        return traj


    return done, state

def main():
    flag = "train"
    parser = rosi_arg_parser()
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm', 'mlp'], default='mlp')
    #parser.add_argument('--model-path', default=os.path.join(logger.get_dir(), 'models'))
    #parser.add_argument('--env', help='environment ID', default='Rosi-v0')
    args = parser.parse_args()
    logger.configure()
    if flag=="train":
        model = train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)
    else:
        #pi = train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)
        env = make_rosenbrock_env(args.env, seed=0)
        env.reset()
        pi = policy_fn("pi", env.observation_space, env.action_space)  # create network
        U.make_session(num_cpu=1).__enter__()
        U.load_state("/Users/romc/Documents/RNN_exploration_learning/baselines/baselines/ppo1/tmp/test")
        ob = np.array([5.0,2.0, 1.0, 100.0, 10.0])
        env.set_state(ob)
        step = 0
        while True: #step < 10:
            #time.sleep(10)
            step += 1
            action = pi.act(stochastic=False, ob=ob)[0]
            #action = [1]
            print(action)
            ob, _, done, _ =  env.step(action)
            print(ob)
            print("")
            time.sleep(1)
            #print(ob[4]*1000)
            #print(done)
            if done:
                print("last")
                print(step)
                print(ob)
                break

if __name__ == '__main__':
    main()
