import numpy as np
import sys
import os

# Ensure server module is discoverable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from server.app import WarehouseEnv, get_action_from_llm

if __name__ == "__main__":
    tasks = ["warehouse-navigation-test-1", "warehouse-navigation-test-2", "warehouse-navigation-test-3"]
    
    for task_name in tasks:
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

        # Normalize score to be strictly between 0 and 1
        score = 0.5 + (total_reward / 200.0)
        score = max(0.01, min(0.99, score))

        print(f"[END] task={task_name} score={score:.4f} steps={env.current_step}", flush=True)