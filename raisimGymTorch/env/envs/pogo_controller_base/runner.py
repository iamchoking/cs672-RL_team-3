from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher
from raisimGymTorch.env.bin.pogo_controller_base import NormalSampler
from raisimGymTorch.env.bin.pogo_controller_base import RaisimGymEnv
from raisimGymTorch.env.RewardAnalyzer import RewardAnalyzer
from io import StringIO
import os
import math
import time
import raisimGymTorch.algo.ppo.module as ppo_module
import raisimGymTorch.algo.ppo.ppo as PPO
import torch.nn as nn
import numpy as np
import torch
import datetime
import argparse

# task specification
task_name = "pogo_trian1"

# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', help='set mode either train or test', type=str, default='train')
parser.add_argument('-w', '--weight', help='pre-trained weight path', type=str, default='')
args = parser.parse_args()
mode = args.mode
weight_path = args.weight

# check if gpu is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../.."

# config
cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))

# create environment from the configuration file
env = VecEnv(RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)))
env.seed(cfg['seed'])

# shortcuts
ob_dim = env.num_obs
act_dim = env.num_acts
num_threads = cfg['environment']['num_threads']

# Training
n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
total_steps = n_steps * env.num_envs

avg_rewards = []

actor = ppo_module.Actor(ppo_module.GRU_MLP_Actor(ob_dim - 3,
                                                  cfg['architecture']['hidden_dim'],
                                                  cfg['architecture']['mlp_shape'],
                                                  act_dim,
                                                  env.num_envs),
                         ppo_module.MultivariateGaussianDiagonalCovariance(act_dim,
                                                                           env.num_envs,
                                                                           5.0,
                                                                           NormalSampler(act_dim),
                                                                           cfg['seed']), device)

critic = ppo_module.Critic(ppo_module.MLP(cfg['architecture']['mlp3_shape'], nn.LeakyReLU, ob_dim, 1), device)

saver = ConfigurationSaver(log_dir=home_path + "/raisimGymTorch/data/"+task_name,
                           save_items=[task_path + "/cfg.yaml", task_path + "/Environment.hpp"])
tensorboard_launcher(saver.data_dir+"/..")  # press refresh (F5) after the first ppo update

ppo = PPO.PPO(actor=actor,
              critic=critic,
              num_envs=cfg['environment']['num_envs'],
              num_transitions_per_env=n_steps,
              num_learning_epochs=32,
              gamma=0.99,
              lam=0.95,
              num_mini_batches=1,
              policy_learning_rate=5e-4,
              value_learning_rate=5e-4,
              lr_scheduler_rate=0.9999079,
              max_grad_norm=0.5,
              device=device,
              log_dir=saver.data_dir,
              shuffle_batch=False,
              entropy_coef=0.005,
              value_loss_coef=0.5,
              )

reward_analyzer = RewardAnalyzer(env, ppo.writer)

def reset():
    env.reset()
    ppo.actor.architecture.init_hidden()


def by_terminate(dones):
    if np.sum(dones) > 0:
        arg_dones = np.argwhere(dones).flatten()
        ppo.actor.architecture.init_by_done(arg_dones)

for update in range(50000):
    start = time.time()
    reset()
    reward_sum = 0
    done_sum = 0
    average_dones = 0.

    if update % cfg['environment']['iteration_per_save'] == 0:
        torch.save({
            'actor_architecture_state_dict': actor.architecture.state_dict(),
            'actor_distribution_state_dict': actor.distribution.state_dict(),
            'critic_architecture_state_dict': critic.architecture.state_dict(),
            'policy_optimizer_state_dict': ppo.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': ppo.value_optimizer.state_dict(),
            'policy_scheduler_state_dict': ppo.policy_scheduler.state_dict(),
            'value_scheduler_state_dict': ppo.value_scheduler.state_dict(),
        }, saver.data_dir + "/full_" + str(update) + '.pt')
        env.save_scaling(saver.data_dir, str(update))

    if update % cfg['environment']['eval_every_n'] == 0:
        print("Visualizing and evaluating the current policy")

        env.turn_on_visualization()
        env.start_video_recording(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "policy_"+str(update)+'.mp4')

        for step in range(n_steps):
            with torch.no_grad():
                frame_start = time.time()
                obs = env.observe(False)
                actions, _ = actor.sample(torch.from_numpy(np.expand_dims(obs, axis=0)).to(device))
                reward, dones = env.step(actions)
                reward_analyzer.add_reward_info(env.get_reward_info())
                frame_end = time.time()
                wait_time = cfg['environment']['control_dt'] - (frame_end-frame_start)
                if wait_time > 0.:
                    time.sleep(wait_time)

        env.stop_video_recording()
        env.turn_off_visualization()

        reward_analyzer.analyze_and_plot(update)
        reset()


    # actual training
    for step in range(n_steps):
        with torch.no_grad():
            obs = env.observe(update < 10000)
            action = ppo.act(np.expand_dims(obs, axis=0))
            reward, dones = env.step(action)
            ppo.step(value_obs=obs, rews=reward, dones=dones)
            done_sum = done_sum + np.sum(dones)
            reward_sum = reward_sum + np.sum(reward)
            by_terminate(dones)

    # take st step to get value obs
    obs = env.observe()
    ppo.update(value_obs=np.expand_dims(obs, axis=0),
               log_this_iteration=update % 10 == 0, update=update)
    average_ll_performance = reward_sum / total_steps
    average_dones = done_sum / total_steps
    avg_rewards.append(average_ll_performance)

    actor.update()
    #actor.distribution.enforce_minimum_std((torch.ones(3)*0.2).to(device)) # this line has hard-coded action dim...

    if update % cfg['environment']['curriculum']['iteration_per_update'] == 0:
        env.curriculum_callback()

    end = time.time()

    print('----------------------------------------------------')
    print('{:>6}th iteration'.format(update))
    print('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(average_ll_performance)))
    print('{:<40} {:>6}'.format("dones: ", '{:0.6f}'.format(average_dones)))
    print('{:<40} {:>6}'.format("time elapsed in this iteration: ", '{:6.4f}'.format(end - start)))
    print('{:<40} {:>6}'.format("fps: ", '{:6.0f}'.format(total_steps / (end - start))))
    print('{:<40} {:>6}'.format("real time factor: ", '{:6.0f}'.format(total_steps / (end - start)
                                                                       * cfg['environment']['control_dt'])))
    print('----------------------------------------------------\n')
