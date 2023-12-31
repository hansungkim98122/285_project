import os
import time
import yaml

from cs285.agents.sac import SACAgent
from cs285.infrastructure.replay_buffer import ReplayBuffer
import cs285.env_configs

import os
import sys
sys.path.append('/'.join(os.getcwd().split('/')[:-2]) + '/TeachMyAgent_modified/')
from LLM.TerrainGen import LLMTerrianGenerator
import time

import gym
from gym import wrappers
import numpy as np
import torch
from cs285.infrastructure import pytorch_util as ptu
import tqdm

from cs285.infrastructure import utils
from cs285.infrastructure.logger import Logger

from scripting_utils import make_logger, make_config

import argparse
import TeachMyAgent.environments


def run_training_loop(config: dict, logger: Logger, args: argparse.Namespace):
    # set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    # make the gym environment
    if args.mode == 'manual':
        env = config["make_env"](agent_body_type='classic_bipedal', movable_creepers=True, max_steps = config["ep_len"])
        eval_env = config["make_env"](agent_body_type='classic_bipedal', movable_creepers=True, max_steps = config["ep_len"])
        render_env = config["make_env"](agent_body_type='classic_bipedal', movable_creepers=True, max_steps = config["ep_len"])

        input_vector = np.array([-0.058,0.912,0.367])
        env.set_environment(input_vector=input_vector, water_level = -100)

        input_vector = np.array([-0.058,0.912,0.367])
        eval_env.set_environment(input_vector=input_vector, water_level = -100)

        input_vector = np.array([-0.058,0.912,0.367])
        render_env.set_environment(input_vector=input_vector, water_level = -100)
    else:
        #Load LLM config
        with open(args.llm_config_file, 'r') as f:
            llm_config = yaml.load(f, Loader=yaml.FullLoader)

        #Initialize the terrain generator (LLM)
        try:
            # llm = LLMTerrianGenerator(llm_config['horizon'], llm_config['top'], llm_config['bottom'], llm_config['model'], llm_config['temperature'], llm_config['sample'], llm_config["smooth_window"])
            llm = LLMTerrianGenerator(llm_config)
            print('LLM successfully Generated')
        except:
            print('ERROR: Could not initialize LLM. Exiting.')

        #Use LLM
        env = config["make_env"](agent_body_type='classic_bipedal', movable_creepers=True, mode='llm', max_steps = config["ep_len"])
        eval_env = config["make_env"](agent_body_type='classic_bipedal', movable_creepers=True, mode='llm', max_steps = config["ep_len"])
        render_env = config["make_env"](agent_body_type='classic_bipedal', movable_creepers=True, mode='llm', max_steps = config["ep_len"])

        ground_y = llm.init_generate(debug=True) #(200,)
        y_terrain = np.vstack((ground_y,ground_y + 100)) #100 is the hardcoded offset for the ceiling

        assert y_terrain.shape == (2,200)

        env.set_terrain(y_terrain, water_level = -100)
        eval_env.set_terrain(y_terrain, water_level = -100)
        render_env.set_terrain(y_terrain, water_level = -100)

    ep_len = config["ep_len"] or env.spec.max_episode_steps
    batch_size = config["batch_size"] or batch_size
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    assert (
        not discrete
    ), "Our actor-critic implementation only supports continuous action spaces. (This isn't a fundamental limitation, just a current implementation decision.)"

    ob_shape = env.observation_space.shape
    ac_dim = env.action_space.shape[0]

    config["agent_kwargs"]['action_range'] = [
        float(env.action_space.low.min()),
        float(env.action_space.high.max())
    ]
    config["agent_kwargs"]['device'] = ptu.device
    
    fps = 20
    # initialize agent
    agent = SACAgent(
        ob_shape,
        ac_dim,
        **config["agent_kwargs"],
    )

    replay_buffer = ReplayBuffer(config["replay_buffer_capacity"])
    train_freq = config['training_freq']
    observation = env.reset()
    
    for step in tqdm.trange(config["total_steps"], dynamic_ncols=True):
        if step < config["random_steps"]:
            action = env.action_space.sample()
        else:
            action = agent.act(observation)
        # Step the environment and add the data to the replay buffer
        next_observation, reward, done, info = env.step(action)
        replay_buffer.insert(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            done=done and not info.get("TimeLimit.truncated", False),
        )

        if done:
            logger.log_scalar(info["episode"]["r"], "train_return", step)
            logger.log_scalar(info["episode"]["l"], "train_ep_len", step)
            observation = env.reset()

            #training logging for debugging
            # train_return = info["episode"]["r"]
            # train_ep_len = info["episode"]["l"]
            # print(f'Step: {step} Train return: {train_return}, ep len:{train_ep_len}')
        else:
            observation = next_observation
        
        
        # Train the agent
        if step >= config["training_starts"]:
            
            for _ in range(train_freq):
                batch = replay_buffer.sample(config['batch_size'])
                
                update_info = agent.update(ptu.from_numpy(batch['observations']),ptu.from_numpy(batch['actions']),ptu.from_numpy(batch['rewards']),ptu.from_numpy(batch['next_observations']),ptu.from_numpy(batch['dones']),step)

            # Logging
            # update_info["actor_lr"] = agent.actor_lr_scheduler.get_last_lr()[0]
            # update_info["critic_lr"] = agent.critic_lr_scheduler.get_last_lr()[0]

            if step % args.log_interval == 0:
                for k, v in update_info.items():
                    logger.log_scalar(v, k, step)
                    logger.log_scalars
                logger.flush()

        # Run evaluation
        if step % args.eval_interval == 0:
            trajectories = utils.sample_n_trajectories(
                eval_env,
                policy=agent,
                ntraj=args.num_eval_trajectories,
                max_length=ep_len,
            )
            returns = [t["episode_statistics"]["r"] for t in trajectories]
            ep_lens = [t["episode_statistics"]["l"] for t in trajectories]

            logger.log_scalar(np.mean(returns), "eval_return", step)
            logger.log_scalar(np.mean(ep_lens), "eval_ep_len", step)
            print(f'Step: {step} Eval return: {np.mean(returns)}')
            if len(returns) > 1:
                logger.log_scalar(np.std(returns), "eval/return_std", step)
                logger.log_scalar(np.max(returns), "eval/return_max", step)
                logger.log_scalar(np.min(returns), "eval/return_min", step)
                logger.log_scalar(np.std(ep_lens), "eval/ep_len_std", step)
                logger.log_scalar(np.max(ep_lens), "eval/ep_len_max", step)
                logger.log_scalar(np.min(ep_lens), "eval/ep_len_min", step)

            if args.num_render_trajectories > 0:
                video_trajectories = utils.sample_n_trajectories(
                    render_env,
                    agent,
                    args.num_render_trajectories,
                    ep_len,
                    render=False,
                )

                logger.log_paths_as_videos(
                    video_trajectories,
                    step,
                    fps=fps,
                    max_videos_to_save=args.num_render_trajectories,
                    video_title="eval_rollouts",
                )
        if args.mode == 'llm' and step % args.llm_feedback_period == 0 and step > 0:
            #LLM feedback
            #Updates the environment with the new terrain provided by the LLM
            llm.llm_feedback(logger, [env, eval_env, render_env], debug=True)
        


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-cfg", type=str, required=True)
    parser.add_argument("--llm_config_file", "-llm_cfg", type=str, default='')
    parser.add_argument("--llm_feedback_period", "-llm_fb", type=int, default=100000)

    parser.add_argument("--eval_interval", "-ei", type=int, default=10000)
    parser.add_argument("--num_eval_trajectories", "-neval", type=int, default=5)
    parser.add_argument("--num_render_trajectories", "-nvid", type=int, default=0)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-g", default=0)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--mode", type=str, default='manual')

    args = parser.parse_args()

    # create directory for logging
    logdir_prefix = ""  # keep for autograder

    config = make_config(args.config_file)
    logger = make_logger(logdir_prefix, config)

    run_training_loop(config, logger, args)


if __name__ == "__main__":
    main()
