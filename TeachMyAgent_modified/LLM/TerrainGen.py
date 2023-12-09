# import hydra
import numpy as np
import json
import logging 
import matplotlib.pyplot as plt
import os
import openai
import re
import subprocess
from pathlib import Path
import shutil
import time 
from collections import defaultdict
from datetime import datetime

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# set logging level to info
logging.basicConfig(level=logging.INFO)

def file_to_string(filename):
    with open(filename, 'r') as file:
        return file.read()
    
def load_tensorboard_logs(path):
    data = defaultdict(list)
    event_acc = EventAccumulator(path)
    event_acc.Reload()  # Load all data written so far

    for tag in event_acc.Tags()["scalars"]:
        events = event_acc.Scalars(tag)
        for event in events:
            data[tag].append(event.value)
    
    return data

class LLMTerrianGenerator:
    
    def __init__(self, cfg: dict):
                #  horizon: int, 
                #  top, 
                #  bottom, 
                #  model: str = "gpt-3.5-turbo", 
                #  temperature: float = 0.5,
                #  sample: int = 4,
                #  smooth_window: int = 1
                 
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("LLMTerrianGenerator initializing...")
        self.horizon = cfg['horizon']
        self.top = cfg['top'] +0.01
        self.bottom = cfg['bottom']-0.01
        self.model = cfg['model'] 
        self.temperature = cfg['temperature']
        self.sample = cfg['sample']
        self.chunk_size = 4
        self.smooth_window = cfg['smooth_window']
        
        openai.api_key = cfg['openaikey']
        # This is to store all the responses.
        self.responses = []

        current_datetime = datetime.now()
        formatted_date = current_datetime.strftime('%d-%m-%Y-%H-%M-%S')
       
        if not os.path.exists('./llmlog'):
            os.makedirs('./llmlog')
        self.message_log = f"llmlog/message_{formatted_date}.txt"
        logging.info(f"Message is logged in: {self.message_log}")
        
        openai.api_key = os.getenv("OPENAI_API_KEY")
    
       # Loading all text prompts
        prompt_dir = './TeachMyAgent_modified/LLM/Prompt'
        self.initial_system = file_to_string(f'{prompt_dir}/initial_system.txt')
        terrain_example = file_to_string(f'{prompt_dir}/terrain_example.txt')
        self.code_output_tip = file_to_string(f'{prompt_dir}/code_output_tip.txt')+terrain_example
        self.code_feedback = file_to_string(f'{prompt_dir}/code_feedback.txt')
        self.initial_user = file_to_string(f'{prompt_dir}/initial_user.txt')
        # reward_signature = file_to_string(f'{prompt_dir}/reward_signature.txt')
        self.policy_feedback = file_to_string(f'{prompt_dir}/policy_feedback.txt')
        self.execution_error_feedback = file_to_string(f'{prompt_dir}/execution_error_feedback.txt')

        self.initial_system = self.initial_system.format(terrain_horizon_length=self.horizon,terrain_bottom=self.bottom,terrain_top=self.top) + self.code_output_tip
        
        self.messages = [{"role": "system", "content": self.initial_system}, {"role": "user", "content": self.initial_user}]

        self.logger.info("LLMTerrianGenerator initialized")
        
        print(self.initial_system)
        
    def _callOpenAI(self,messages,debug:bool = False):
    
        response_cur = None
        total_samples = 0
        total_token = 0
        total_completion_token = 0
        
        logging.info(f"Generating samples with {self.model}")
        total_samples = 0
  
        for attempt in range(10):
            try:
                response_cur = openai.ChatCompletion.create(
                    model=self.model,
                    messages=messages,
                    # temperature=self.temperature,
                    # n=self.chunk_size
                )
                total_samples += self.chunk_size
                logging.info("LLM call succeeded!")
                break
            except Exception as e:
                if attempt >= 10:
                    self.chunk_size = max(int(self.chunk_size / 2), 1)
                    print("Current Chunk Size", self.chunk_size)
                logging.info(f"Attempt {attempt+1} failed with error: {e}")
                time.sleep(1)

        if response_cur is None:
            logging.info("Code terminated due to too many failed attempts!")
            exit()

        self.responses.extend(response_cur['choices'])
        prompt_tokens = response_cur['usage']['prompt_tokens']
        total_completion_token += response_cur['usage']['completion_tokens']
        total_token += response_cur['usage']['total_tokens']


        # Logging Token Information
        logging.info(f"Prompt Tokens: {prompt_tokens}, Completion Tokens: {total_completion_token}, Total Tokens: {total_token}")
    
        self._log_messge(response_cur)
        
        ret = self.responses[0]['message']['content']
        if debug:
            print(ret)
        ret = list(map(float,ret.split('[')[-1].split(']')[0].split(', ')))
        ret = ret+ret[-1]*100
        ret = ret[:self.horizon]
        
        if debug:
            plt.plot(ret)
            plt.ylim(-20,20)
        ret = LLMTerrianGenerator.smooth_array(ret, self.smooth_window)
        logging.info(f"The generated terrain is {ret}")
        if debug:
            plt.plot(ret)
            plt.ylim(-20,20)
            plt.savefig('terrain.png')
        assert len(ret) == self.horizon, "The length of the generated terrain is not correct!"
        return ret
    
    def _log_messge(self,message,log_format='json'):
        logging.info(f"Logging Message...")
        if log_format == 'str':
            assert isinstance(message,str), "message must be a string!"
            with open(self.message_log, 'a') as file:
                file.write(message)
                file.write('\n')
                
        elif log_format == 'json':
            assert isinstance(message,dict), "message must be a dict!"
            with open(self.message_log, 'a') as file:
                json.dump(message, file, indent=4)
        else:
            raise NotImplementedError(f"log_format {log_format} not implemented!")
        
    def init_generate(self,debug:bool = False):
        logging.info("Generating initial terrain...")
        message = self._callOpenAI(self.messages,debug)
        return message

    def iter_generate(self,tensorboard_logdir: str = None,debug:bool = False):
        logging.info("Generating terrain with updated metric...")
        self._update_message(tensorboard_logdir)
        message = self._callOpenAI(self.messages,debug) 
        return message
    
    def _update_message(self,tensorboard_logdir: str = None):

        assert tensorboard_logdir is not None, "tensorboard_logdir must be provided!"

        # code_feedbacks = []
        # contents = []
        # successes = []
        # reward_correlations = []

        tensorboard_logs = load_tensorboard_logs(tensorboard_logdir)
        max_iterations = np.array(tensorboard_logs['critic_loss']).shape[0]
        epoch_freq = max(int(max_iterations // 10), 1)

        content += self.policy_feedback.format(epoch_freq=epoch_freq)

        # # Compute Correlation between Human-Engineered and GPT Rewards
        # if "gt_reward" in tensorboard_logs and "gpt_reward" in tensorboard_logs:
        #     gt_reward = np.array(tensorboard_logs["gt_reward"])
        #     gpt_reward = np.array(tensorboard_logs["gpt_reward"])
        #     reward_correlation = np.corrcoef(gt_reward, gpt_reward)[0, 1]
        #     reward_correlations.append(reward_correlation)

        # Add reward components log to the feedback
        for metric in tensorboard_logs:
            if "/" not in metric:
                metric_cur = ['{:.2f}'.format(x) for x in tensorboard_logs[metric][::epoch_freq]]
                metric_cur_max = max(tensorboard_logs[metric])
                metric_cur_mean = sum(tensorboard_logs[metric]) / len(tensorboard_logs[metric])
                # if "consecutive_successes" == metric:
                #     successes.append(metric_cur_max)
                # metric_cur_min = min(tensorboard_logs[metric])
                # if metric != "gt_reward" and metric != "gpt_reward":
                #     if metric != "consecutive_successes":
                #         metric_name = metric 
                #     else:
                #         metric_name = "task_score"
                #     content += f"{metric_name}: {metric_cur}, Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f} \n"                    
                # else:
                #     # Provide ground-truth score when success rate not applicable
                #     if "consecutive_successes" not in tensorboard_logs:
                #         content += f"ground-truth score: {metric_cur}, Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f} \n"                    
                metric_name = metric
                content += f"{metric_name}: {metric_cur}, Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f} \n"                    

        # code_feedbacks.append(self.code_feedback)
        content += self.code_feedback  
        content += self.code_output_tip

        if len(self.messages) == 2:
            self.messages += [{"role": "assistant", "content": self.responses[0]["message"]["content"]}]
            self.messages += [{"role": "user", "content": content}]
        else:
            assert len(self.messages) == 4
            self.messages[-2] = {"role": "assistant", "content": self.responses[0]["message"]["content"]}
            self.messages[-1] = {"role": "user", "content": content}

        logging.info("Message Updated with the newest metric!")
    
    @staticmethod
    def smooth_array(arr, window_size:int):
        if window_size < 1:
            raise ValueError("Window size must be at least 1")
        if window_size > len(arr):
            raise ValueError("Window size is too large")
            
        # Create a simple moving average kernel
        kernel = np.ones(window_size) / window_size

        # Apply the convolution
        smoothed_arr = np.convolve(arr, kernel, mode='valid')

        return smoothed_arr


if __name__ == '__main__':
    horizon = 200
    top = 20
    bottom = -20
    model = "gpt-3.5-turbo"
    temperature = 0.5
    sample = 4
    smooth_window = 25

    llm = LLMTerrianGenerator(horizon, top, bottom, model, temperature, sample, smooth_window)
    terrain = llm.init_generate(debug=True)

    print(terrain)
