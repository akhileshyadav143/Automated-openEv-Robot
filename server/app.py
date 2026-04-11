import os
import numpy as np
from openai import OpenAI
from flask import Flask, request, jsonify

class WarehouseEnv:
    def __init__(self):
        self.grid = np.array([
            [0, 0, 0, 1, 0],
            [0, 1, 0, 1, 3],
            [0, 1, 0, 0, 0],
            [2, 0, 1, 1, 0],
            [0, 0, 0, 0, 0]
        ])
        self.start_pos = [0, 0]
        self.agent_pos = list(self.start_pos)
        self.has_item = False
        self.max_steps = 50
        self.current_step = 0

    def reset(self):
        self.agent_pos = list(self.start_pos)
        self.has_item = False
        self.current_step = 0
        self.grid[3, 0] = 2 
        return self._get_obs()

    def _get_obs(self):
        return f"Grid:\n{self.grid}\nAgent Position: {self.agent_pos}\nHas Item: {self.has_item}"

    def step(self, action):
        self.current_step += 1
        row, col = self.agent_pos
        next_row, next_col = row, col

        if action == 0: next_row -= 1     
        elif action == 1: next_row += 1   
        elif action == 2: next_col -= 1   
        elif action == 3: next_col += 1   

        reward = -1.0 
        done = False
        info = "null"

        if (next_row < 0 or next_row >= self.grid.shape[0] or 
            next_col < 0 or next_col >= self.grid.shape[1] or 
            self.grid[next_row, next_col] == 1):
            reward = -15.0 
            info = "Hit boundary"
        else:
            self.agent_pos = [next_row, next_col]
            if self.grid[next_row, next_col] == 2 and not self.has_item:
                self.has_item = True
                self.grid[next_row, next_col] = 0 
            if self.grid[next_row, next_col] == 3 and self.has_item:
                reward = 100.0 
                done = True

        if self.current_step >= self.max_steps: done = True
        return self._get_obs(), float(reward), done, info

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/hf-inference/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

client = None
if HF_TOKEN:
    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    except:
        pass

def get_action_from_llm(obs):
    if not client: return 1
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
            model=MODEL_NAME, messages=[{"role": "user", "content": prompt}], max_tokens=10
        )
        action_num = ''.join(filter(str.isdigit, response.choices[0].message.content.strip()))
        return int(action_num[0]) if action_num else 1
    except:
        return 1

app = Flask(__name__)
env = WarehouseEnv()

@app.route('/reset', methods=['POST'])
def reset_endpoint():
    global env
    return jsonify({"observation": env.reset()})

@app.route('/step', methods=['POST'])
def step_endpoint():
    global env
    obs = env._get_obs()
    action = get_action_from_llm(obs)
    obs, reward, done, error_msg = env.step(action)
    return jsonify({"observation": obs, "reward": reward, "done": done, "error": error_msg})

@app.route('/', methods=['GET'])
def health():
    return "API is Ready for Hackathon Grader"

# Grader ko specifically ye 'main' function chahiye tha
def main():
    app.run(host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()