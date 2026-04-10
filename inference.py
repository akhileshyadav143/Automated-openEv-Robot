import os
from openai import OpenAI
from warehouse_env import WarehouseEnv

# 1. Required Environment Variables from Hackathon Guidelines
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# 2. Must use the OpenAI Client
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

def get_action_from_llm(obs):
    prompt = f"""
    You are an automated warehouse robot. Goal: Navigate a maze, pick up an item (2), and drop it at destination (3).
    Grid values: 0=path, 1=wall, 2=item, 3=destination.
    Actions: 0 (Up), 1 (Down), 2 (Left), 3 (Right).
    
    Current State:
    {obs}
    
    Respond with ONLY a single integer representing your chosen action.
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )
        action_str = response.choices[0].message.content.strip()
        action = int(action_str)
        return action, action_str
    except Exception as e:
        return 1, "invalid_action"

def run_inference():
    env = WarehouseEnv()
    obs = env.reset()
    
    task_name = "warehouse-navigation"
    benchmark = "openenv-grid"
    
    steps = 0
    rewards_history = []
    success = False
    
    # 3. Exactly formatted START line
    print(f"[START] task={task_name} env={benchmark} model={MODEL_NAME}")
    
    try:
        done = False
        while not done:
            action, action_str = get_action_from_llm(obs)
            obs, reward, done, error_msg = env.step(action)
            
            steps += 1
            rewards_history.append(reward)
            
            # Format requirements: boolean must be lowercase, reward to 2 decimal places
            done_str = "true" if done else "false"
            if reward == 100.0:
                success = True
                
            # 4. Exactly formatted STEP line immediately after env.step()
            print(f"[STEP] step={steps} action={action_str} reward={reward:.2f} done={done_str} error={error_msg}")
            
    except Exception as e:
        pass # The [END] line is always emitted even on exception
        
    finally:
        success_str = "true" if success else "false"
        rewards_str = ",".join([f"{r:.2f}" for r in rewards_history]) if rewards_history else "0.00"
        
        # 5. Exactly formatted END line
        print(f"[END] success={success_str} steps={steps} rewards={rewards_str}")

if __name__ == "__main__":
    run_inference()