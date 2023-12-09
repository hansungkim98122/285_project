export OPENAI_API_KEY= "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"


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

SAC Agent
```
python cs285/scripts/run_sac.py -cfg experiments/sac/parkour.yaml -nvid 1
```

PPO Agent
```
python cs285/scripts/run_ppo.py -cfg experiments/ppo/parkour.yaml -nvid 1
```


