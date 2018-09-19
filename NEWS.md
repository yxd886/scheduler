SPECIFICATION OF THIS VERSION OF CODE

v5

supervised learning + reinforcement learning
synthetic trace
inject samples when workers > a randomly genearted number
Supervised learning: same performance as DRF
Reinforcement learning: continue to improve by about 20%

Adding PS+worker for rl_env.py and drf_env.py
fixing some bugs in rl_env.py

Refactor the code: create a base class scheduler_base.py as parent class for all schedulers
Support real speed trace

Change policy NN architecture with shortcut conn, do not know how to handle value network
