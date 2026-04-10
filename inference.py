import os
from openai import OpenAI
from flask import Flask, request, jsonify
from warehouse_env import WarehouseEnv

app = Flask(__name__)

API_BASE_URL = "https://api-inference.huggingface.co/v1/"
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
HF_TOKEN = os.getenv("HF_TOKEN")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

# Global environment variable
env = WarehouseEnv()
task_name = "warehouse-navigation"
benchmark = "openenv-grid"
steps = 0
rewards_history = []
success = False

def get_action_from_llm(obs):
    prompt = f"""
    You are an automated warehouse robot. Goal: Navigate a maze, pick up an item (2), and drop it at destination (3).
    Grid values: 0=path, 1=wall, 2=item, 3=destination.
    Actions: 0 (Up), 1 (Down), 2 (Left), 3 (Right).
    
    Current State:
    {obs}
    
    Respond with ONLY a single number (0, 1, 2, or 3).
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10
        )
        action_str = response.choices[0].message.content.strip()
        action_num = ''.join(filter(str.isdigit, action_str))
        if action_num:
            return int(action_num[0])
        else:
            return 1
    except Exception as e:
        return 1

# Endpoint 1: Grader yahan Reset request bhejega
@app.route('/reset', methods=['POST'])
def reset():
    global env, steps, rewards_history, success
    obs = env.reset()
    steps = 0
    rewards_history = []
    success = False
    
    print(f"[START] task={task_name} env={benchmark} model={MODEL_NAME}")
    return jsonify({"observation": obs})

# Endpoint 2: Grader yahan Step request bhejega
@app.route('/step', methods=['POST'])
def step():
    global env, steps, rewards_history, success
    
    obs = env._get_obs()
    action = get_action_from_llm(obs)
    
    obs, reward, done, error_msg = env.step(action)
    
    steps += 1
    rewards_history.append(reward)
    
    done_str = "true" if done else "false"
    if reward == 100.0:
        success = True
        
    print(f"[STEP] step={steps} action={action} reward={reward:.2f} done={done_str} error={error_msg}")
    
    if done:
        success_str = "true" if success else "false"
        rewards_str = ",".join([f"{r:.2f}" for r in rewards_history]) if rewards_history else "0.00"
        print(f"[END] success={success_str} steps={steps} rewards={rewards_str}")
        
    return jsonify({
        "observation": obs,
        "reward": float(reward),
        "done": done,
        "error": error_msg
    })


@app.route('/', methods=['GET'])
def health():
    return "Warehouse Bot API is Running"

if __name__ == "__main__":
    # Server ko start karna
    app.run(host="0.0.0.0", port=7860)