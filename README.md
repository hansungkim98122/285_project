Installing TeachMyAgent Package (Optional) and necessary libraries

Copy repo:
```
git clone https://github.com/hansungkim98122/285_project.git
```

Installing the Environment:
```
conda create -f environment.yml
conda activate cur_llm
```
Installing the CS285 Package:
```
cd 285_project
pip install -e .
pip install -r requirements.txt
```
(Optional) If you wish to run Parkour-v0 experiments run:
```
cd TeachMyAgent_modified
pip install -e .
pip install pyglet==1.5.27
cd ..
```

Run these following tests to ensure that the environment is configured correctly:

```
python test_imports.py
```
(Optional) If you wish to run Parkour-v0 experiments run:
```
python test_env.py
```

## Running the experiments

# ACL-LLM:
```
python cs285/scripts/run_dqn.py -cfg experiments/dqn/lunarlander_doubleq.yaml --mode llm -llm_cfg experiments/llm_config.yaml -llm_fb 50000 --log_interval 20 -llm_n_env 10
python cs285/scripts/run_dqn.py -cfg experiments/dqn/lunarlander_doubleq.yaml --mode llm -llm_cfg experiments/llm_config.yaml -llm_fb 100000 --log_interval 20 -llm_n_env 10
python cs285/scripts/run_dqn.py -cfg experiments/dqn/lunarlander_doubleq.yaml --mode llm -llm_cfg experiments/llm_config.yaml -llm_fb 300000 --log_interval 20 -llm_n_env 10
```
To replicate the results in our report, you don't have to run this as the pre-trained model is uploaded:


# Baseline (argument -llm_fb must match with what was used to generate <llm_message_name>):
Example:
```
python cs285/scripts/run_dqn_baseline.py -cfg experiments/dqn/lunarlander_doubleq.yaml -llm_lfp llmlog/<llm_message_name> --log_interval 20  -llm_fb 100000
```
To replicate the results in our report:
```
python cs285/scripts/run_dqn_baseline.py -cfg experiments/dqn/lunarlander_doubleq.yaml -llm_lfp llmlog/message_13-12-2023-10-17-22.txt --log_interval 20 -llm_fb 50000
python cs285/scripts/run_dqn_baseline.py -cfg experiments/dqn/lunarlander_doubleq.yaml -llm_lfp llmlog/message_12-12-2023-21-39-48.txt --log_interval 20 -llm_fb 100000
python cs285/scripts/run_dqn_baseline.py -cfg experiments/dqn/lunarlander_doubleq.yaml -llm_lfp llmlog/message_12-12-2023-21-26-20.txt --log_interval 20 -llm_fb 300000
```

# Evaluation:
Example:
```
python evaluate.py -cfg experiments/dqn/lunarlander_doubleq.yaml -neval 10 -nte 100 -bmd <baseline_model_dir> -lmd <llm_model_dir>
```
To replicate the results in our report:
```
python evaluate.py -cfg experiments/dqn/lunarlander_doubleq.yaml -neval 10 -nte 100 -bmd model/baseline_50k_13-12-2023_10-57-18_dqn_model.pt -lmd model/llm_50k_13-12-2023_10-17-22_dqn_model.pt
python evaluate.py -cfg experiments/dqn/lunarlander_doubleq.yaml -neval 10 -nte 100 -bmd model/baseline_100k_12-12-2023_23-58-01_dqn_model.pt -lmd model/llm_100k_12-12-2023_21-39-48_dqn_model.pt
python evaluate.py -cfg experiments/dqn/lunarlander_doubleq.yaml -neval 10 -nte 100 -bmd model/baseline_300k_13-12-2023_01-33-38_dqn_model.pt -lmd model/llm_300k_12-12-2023_21-26-20_dqn_model.pt
```

## Extras (LunarLander-V2 environment):
# Experiments 
DQN Agent:
```
python cs285/scripts/run_dqn.py -cfg experiments/dqn/lunarlander_doubleq.yaml --log_interval 15
```
CQL Agent:
```
python cs285/scripts/run_cql.py -cfg experiments/dqn/lunarlander_cql.yaml --log_interval 15
```

## Extras (Parkour-v0 Environment):
# Experiments 

SAC Agent (sac.py):

```
python cs285/scripts/run_sac_new.py -cfg experiments/sac_new/parkour.yaml --log_interval 15
```

PPO Agent (ppo_agent.py)
```
python cs285/scripts/run_ppo.py -cfg experiments/ppo/parkour.yaml -nvid 1 --log_interval 15
```

# Automatic Curriculum Learning via LLM (Parkour-v0 Environment):

SAC Agent
```
python cs285/scripts/run_sac_new.py -cfg experiments/sac_new/parkour.yaml -nvid 1 --log_interval 15 --mode llm -llm_cfg experiments/llm_config.yaml -llm_fb 100000
```

PPO Agent
```
python cs285/scripts/run_ppo.py -cfg experiments/ppo/parkour.yaml -nvid 1 --log_interval 15 --mode llm -llm_cfg experiments/llm_config.yaml -llm_fb 100000
```
