export OPENAI_API_KEY="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"


Installing TeachMyAgent Package and necessary libraries

Copy repo:
```
git clone https://github.com/hansungkim98122/285_project.git
```

Python 3.10
Installing the Environment:
```
conda create --name teachMyAgent
conda activate teachMyAgent
pip install -e .
```
Installing the CS285 Package:
```
pip install pyglet==1.5.27
```
```
cd 285_project
pip install -e .
pip install -r requirements.txt
```

Run these following tests to ensure that the environment is configured correctly:

```
python test_imports.py
```

```
python test_env.py
```

# Run Experiments

Manual

New SAC Agent (sac.py):

```
python cs285/scripts/run_sac_new.py -cfg experiments/sac_new/parkour.yaml --log_interval 15
python cs285/scripts/run_sac_new.py -cfg experiments/sac_new/parkour.yaml --log_interval 15
```

SAC Agent (soft_actor_critic.py)
```
python cs285/scripts/run_sac.py -cfg experiments/sac/parkour.yaml --log_interval 15
python cs285/scripts/run_sac.py -cfg experiments/sac/parkour.yaml -nvid 1 --log_interval 15
```

PPO Agent (ppo_agent.py)
```
python cs285/scripts/run_ppo.py -cfg experiments/ppo/parkour.yaml -nvid 1 --log_interval 15
```

Automatic Curriculum Learning via LLM:

SAC Agent
```
python cs285/scripts/run_sac.py -cfg experiments/sac/parkour.yaml -nvid 1 --log_interval 15 --mode llm -llm_cfg experiments/llm_config.yaml
```

PPO Agent
```
python cs285/scripts/run_ppo.py -cfg experiments/ppo/parkour.yaml -nvid 1 --log_interval 15 --mode llm -llm_cfg experiments/llm_config.yaml
```


