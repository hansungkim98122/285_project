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

# set logging level to info
logging.basicConfig(level=logging.DEBUG)

def file_to_string(filename):
    with open(filename, 'r') as file:
        return file.read()

class LLMTerrianGenerator:
    
    def __init__(self, 
                 horizon: int, 
                 top: int, 
                 bottom: int, 
                 model: str = "gpt-3.5-turbo", 
                 temperature: float = 0.5,
                 sample: int = 1,
                 chunk_size: int = 4,
                 ):
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("LLMTerrianGenerator initializing...")
        self.horizon = horizon
        self.top = top
        self.bottom = bottom
        self.model = model 
        self.temperature = temperature
        self.sample = sample
        self.chunk_size = chunk_size
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

        self.initial_system = self.initial_system.format(terrain_horizon_length=self.horizon,terrain_bottom=self.bottom,terrain_top=self.top) + self.code_output_tip
        self.messages = [{"role": "system", "content": self.initial_system}, {"role": "user", "content": self.initial_user}]

        self.logger.info("LLMTerrianGenerator initialized")
        
        # print(initial_system)
        
    def _callOpenAI(self,messages):
        
        responses = []
        logging.info(f"Generating {self.sample} samples with {self.model}")
        total_samples = 0
        while True:
            # if total_samples >= self.sample:
            #     logging.info("Too many!")
            #     break
            
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

            responses.extend(response_cur["choices"])
            prompt_tokens = response_cur["usage"]["prompt_tokens"]
            total_completion_token += response_cur["usage"]["completion_tokens"]
            total_token += response_cur["usage"]["total_tokens"]

        if self.sample == 1:
            logging.info(f"GPT Output:\n " + responses[0]["message"]["content"] + "\n")

        # Logging Token Information
        logging.info(f"Prompt Tokens: {prompt_tokens}, Completion Tokens: {total_completion_token}, Total Tokens: {total_token}")
        
    
    def init_generate(self):
        self._callOpenAI(self.messages)
    
    def iter_generate(self):
        pass  
    
    
if __name__ == '__main__':
    llmGen = LLMTerrianGenerator(1000,200,-100)
    llmGen.init_generate()