import os
import gymnasium as gym
import torch
import numpy as np
from model import PolicyNet

if __name__ == '__main__':
    #Create the environment
    env = gym.make('BipedalWalker-v3', render_mode='human')
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]

    #Create the policy net
    policy_net = PolicyNet(s_dim, a_dim)
    print(policy_net)

    #Load the models
    save_dir = './save'
    best_model_path = os.path.join(save_dir, "best_model.pt")
    model_path = os.path.join(save_dir, "model.pt")

    checkpoint_path = best_model_path

    if checkpoint_path is None:
        print('ERROR: No model saved')
        exit(1)

    print(f"Loading the model from {checkpoint_path} ... ", end="")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    policy_net.load_state_dict(checkpoint["PolicyNet"])
    print("Done.")

    #Run an episode using the policy net
    with torch.no_grad():
        reset_out = env.reset()
        state = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        total_reward = 0
        length = 0

        while True:
            env.render()
            state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            action = policy_net.choose_action(state_tensor, deterministic=True).cpu().numpy()
            step_out = env.step(action[0])

            if len(step_out) == 5:
                state, reward, terminated, truncated, info = step_out
                done = terminated or truncated
            else:
                state, reward, done, info = step_out

            total_reward += reward
            length += 1

            if done:
                print("[Evaluation] Total reward = {:.6f}, length = {:d}".format(total_reward, length), flush=True)
                break

    env.close()
