import time
import argparse
import yaml
from cs285.agents.dqn_agent import DQNAgent
import cs285.env_configs

import os
import sys
import time

import gym
from gym import wrappers
import numpy as np
import torch
import pdb
from cs285.infrastructure import pytorch_util as ptu
import tqdm

from cs285.infrastructure import utils
from cs285.infrastructure.logger import Logger
from cs285.scripts.scripting_utils import make_logger, make_config

MAX_NVIDEO = 2


def run_evaluation(config: dict, logger: Logger, args: argparse.Namespace):
    # set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    #Initialize an env
    enable_wind = True
    gravity = np.random.uniform(-11.99,-0.01)
    wind_power = np.random.uniform(0.01,20.00)
    turbulence_power = np.random.uniform(0.01,1.99)

    eval_env = config["make_env"](gravity=gravity, enable_wind=enable_wind, wind_power=wind_power, turbulence_power=turbulence_power)
    discrete = isinstance(eval_env.action_space, gym.spaces.Discrete)

    assert discrete, "DQN only supports discrete action spaces"

    #Load the agents
    base_agent = DQNAgent(
        eval_env.observation_space.shape,
        eval_env.action_space.n,
        **config["agent_kwargs"],
    )

    base_agent.load(args.base_model_dir)

    llm_agent = DQNAgent(
        eval_env.observation_space.shape,
        eval_env.action_space.n,
        **config["agent_kwargs"],
    )

    llm_agent.load(args.llm_model_dir)

    ep_len = eval_env.spec.max_episode_steps
    agent_list = [base_agent, llm_agent] 
    for env_ind in tqdm.tqdm(range(args.num_test_env)):
        #Sample the environment parameters and update
        gravity = np.random.uniform(-11.99,-0.01)
        wind_power = np.random.uniform(0.01,20.00)
        turbulence_power = np.random.uniform(0.01,1.99)
        eval_env.gravity, eval_env.wind_power, eval_env.turbulence_power = gravity, wind_power, turbulence_power

        for agent_id, agent in enumerate(agent_list):
            # Evaluate
            trajectories = utils.sample_n_trajectories(
                eval_env,
                agent,
                args.num_eval_trajectories,
                ep_len,
            )
            returns = [t["episode_statistics"]["r"] for t in trajectories]
            ep_lens = [t["episode_statistics"]["l"] for t in trajectories]

            logger.log_scalar(np.mean(returns), "eval_return_" + str(agent_id), env_ind)
            logger.log_scalar(np.mean(ep_lens), "eval_ep_len_"+ str(agent_id), env_ind)
            print(f'Step: {env_ind} Eval return: {np.mean(returns)}')
            if len(returns) > 1:
                logger.log_scalar(np.std(returns), "eval_" + str(agent_id) +"/return_std", env_ind)
                logger.log_scalar(np.max(returns), "eval_" + str(agent_id) +"/return_max", env_ind)
                logger.log_scalar(np.min(returns), "eval_" + str(agent_id) +"/return_min", env_ind)
                logger.log_scalar(np.std(ep_lens), "eval_" + str(agent_id) +"/ep_len_std", env_ind)
                logger.log_scalar(np.max(ep_lens), "eval_" + str(agent_id) +"/ep_len_max", env_ind)
                logger.log_scalar(np.min(ep_lens), "eval_" + str(agent_id) +"/ep_len_min", env_ind)

                # if args.num_render_trajectories > 0:
                #     video_trajectories = utils.sample_n_trajectories(
                #         render_env,
                #         agent,
                #         args.num_render_trajectories,
                #         ep_len,
                #         render=True,
                #     )

                #     logger.log_paths_as_videos(
                #         video_trajectories,
                #         step,
                #         fps=fps,
                #         max_videos_to_save=args.num_render_trajectories,
                #         video_title="eval_rollouts",
                #     )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-cfg", type=str, required=True)

    parser.add_argument("--num_eval_trajectories", "-neval", type=int, default=5)
    parser.add_argument("--num_test_env", "-nte", type=int, default=5)
    parser.add_argument("--num_render_trajectories", "-nvid", type=int, default=0)

    parser.add_argument("--base_model_dir", "-bmd", type=str, required=True)
    parser.add_argument("--llm_model_dir", "-lmd", type=str, required=True)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-g", default=0)
    parser.add_argument("--log_interval", type=int, default=1)

    args = parser.parse_args()

    # create directory for logging
    logdir_prefix = "eval_"

    config = make_config(args.config_file)
    logger = make_logger(logdir_prefix, config)

    run_evaluation(config, logger, args)


if __name__ == "__main__":
    main()
