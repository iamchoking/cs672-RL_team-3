from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import pogo_controller_base
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
import raisimGymTorch.algo.ppo.module as ppo_module
import os
import math
import time
import torch
import argparse
import pygame
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    print(f"[torch] cuda:{torch.cuda.device_count()} detected.")

# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-w', '--weight', help='trained weight path', type=str, default='')
args = parser.parse_args()

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../.."

# config
cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))

# create environment from the configuration file
cfg['environment']['num_envs'] = 1
cfg['environment']['render'] = True
cfg['environment']['curriculum']['cmd_decay_factor'] = 1

env = VecEnv(pogo_controller_base.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])

# shortcuts
ob_dim = env.num_obs
act_dim = env.num_acts


weight_path = args.weight
iteration_number = weight_path.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0]
weight_dir = weight_path.rsplit('/', 1)[0] + '/'
command = np.zeros(3, dtype=np.float32)


if weight_path == "":
    print("Can't find trained weight, please provide a trained weight with --weight switch\n")
else:
    print("Loaded weight from {}\n".format(weight_path))

    pygame.init()
    pygame.joystick.init()
    joysticks = [pygame.joystick.Joystick(i) for i in range(pygame.joystick.get_count())]
    for joystick in joysticks:
        print("detected" + joystick.get_name())

    n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
    total_steps = n_steps * 1
    start_step_id = 0

    print("Visualizing and evaluating the policy: ", weight_path)
    loaded_graph = ppo_module.GRU_MLP_Actor(ob_dim - 3,
                                            cfg['architecture']['hidden_dim'],
                                            cfg['architecture']['mlp_shape'],
                                            act_dim,
                                            env.num_envs,
                                            device)
    loaded_graph.load_state_dict(torch.load(weight_path, map_location=device)['actor_architecture_state_dict'])
    loaded_graph.init_hidden()

    env.load_scaling(weight_dir, int(iteration_number))
    env.turn_on_visualization()

    try:
        running = True
        while running:
            frame_start = time.time()
            for event in pygame.event.get():  # User did something.
                if event.type == pygame.JOYBUTTONDOWN:  # If user clicked close.
                    if event.button == 1:
                        env.reset()
                        print("env reset")
                    elif event.button == 4:
                        print("Exiting loop")
                        running = False
                        break
            if not running:
                break

            if len(joysticks) > 0:
                command[0] = -3 * joysticks[0].get_axis(1)
                command[2] = -2 * joysticks[0].get_axis(3)
                if(command[0] < -1.0):
                    command[0] = -1.0

            env.setCommand(command)
            obs = env.observe(False)
            with torch.no_grad():
                obs_tensor = torch.from_numpy(np.expand_dims(obs, axis=0)).to(device)
                action_ll = loaded_graph.forward(obs_tensor).squeeze(dim=0)
                env.step(action_ll.cpu().numpy().astype(np.float32))

            frame_end = time.time()
            wait_time = cfg['environment']['control_dt'] - (frame_end-frame_start)
            if wait_time > 0.:
                time.sleep(wait_time)

    except KeyboardInterrupt:
        print("Loop exited on button press")

    finally:
        env.turn_off_visualization()
