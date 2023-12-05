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


def file_to_string(filename):
    with open(filename, 'r') as file:
        return file.read()

class LLMTerrianGenerator:
    
    def __init__(self, horizon: int, top: int, bottom: int, model: str = "gpt-3.5-turbo"):
        self.logger = logging.getLogger(__name__)
        self.logger.info("LLMTerrianGenerator initializing...")
        self.horizon = horizon
        self.top = top
        self.bottom = bottom
        self.model = model 
        
        openai.api_key = os.getenv("OPENAI_API_KEY")
    
       # Loading all text prompts
        prompt_dir = 'TeachMyAgent_modified/LLM/Prompt'
        self.initial_system = file_to_string(f'{prompt_dir}/initial_system.txt')
        self.code_output_tip = file_to_string(f'{prompt_dir}/code_output_tip.txt')
        self.code_feedback = file_to_string(f'{prompt_dir}/code_feedback.txt')
        self.initial_user = file_to_string(f'{prompt_dir}/initial_user.txt')
        # reward_signature = file_to_string(f'{prompt_dir}/reward_signature.txt')
        self.policy_feedback = file_to_string(f'{prompt_dir}/policy_feedback.txt')
        self.execution_error_feedback = file_to_string(f'{prompt_dir}/execution_error_feedback.txt')

        self.initial_system = initial_system.format(terrain_horizon_length=self.horizon,terrain_bottom=self.bottom,terrain_top=self.top) + code_output_tip
        self.initial_user = initial_user
        self.messages = [{"role": "system", "content": initial_system}, {"role": "user", "content": initial_user}]

        self.logger.info("LLMTerrianGenerator initialized")
        
        print(initial_system)
    
    def init_generate(self):
        
        response_cur = openai.ChatCompletion.create(
                model=self.model,
                messages=self.messages,
                temperature=cfg.temperature,
                n=chunk_size
            )
            total_samples += chunk_size 
    
    def iter_generate(self):
        pass  
    
    
if __name__ == '__main__':
    llmGen = LLMTerrianGenerator(100,200,-100)