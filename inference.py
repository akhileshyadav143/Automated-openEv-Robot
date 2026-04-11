import numpy as np
import sys
import os

# Ensure server module is discoverable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from server.app import WarehouseEnv, get_action_from_llm

if __name__ == "__main__":
    task_name = "warehouse-navigation"
    print(f"[START] task={task_name}", flush=True)
    
    env = WarehouseEnv()
    env.reset()
    
    total_reward = 0.0
    done = False
    
    while not done and env.current_step < env.max_steps:
        obs = env._get_obs()
        action = get_action_from_llm(obs)
        obs, reward, done, error_msg = env.step(action)
        
        total_reward += reward
        print(f"[STEP] step={env.current_step} reward={reward}", flush=True)

    print(f"[END] task={task_name} score={total_reward} steps={env.current_step}", flush=True)