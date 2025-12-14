import os
import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
from envrunner import EnvRunner
from model import PolicyNet, ValueNet
from agent import PPO
from torch.utils.tensorboard import SummaryWriter

#Run an episode using the policy net
def play(policy_net):
    render_env = gym.make('BipedalWalker-v3')

    with torch.no_grad():
        state, _ = render_env.reset()
        total_reward = 0
        length = 0

        while True:
            render_env.render()
            state_tensor = torch.tensor(np.expand_dims(state, axis=0), dtype=torch.float32, device='cpu')
            action = policy_net.choose_action(state_tensor, deterministic=True).cpu().numpy()
            state, reward, done, truncated, info = render_env.step(action[0])
            total_reward += reward
            length += 1

            if done:
                print("[Evaluation] Total reward = {:.6f}, length = {:d}".format(total_reward, length), flush=True)
                break

    render_env.close()

#Train the policy net & value net using the agent
def train(env, runner, policy_net, value_net, agent, max_episode=5000):
    mean_total_reward = 0
    mean_length = 0
    best_return = -np.inf
    save_dir = './save'

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    log_dir = os.path.join(save_dir, "logs")
    # 清理旧的TensorBoard日志
    import shutil
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    for i in range(max_episode):
        #Run and episode to collect data
        with torch.no_grad():
            mb_states, mb_actions, mb_old_a_logps, mb_values, mb_returns, mb_rewards = runner.run(env, policy_net, value_net)
            mb_advs = mb_returns - mb_values
            mb_advs = (mb_advs - mb_advs.mean()) / (mb_advs.std() + 1e-6)
        
        #Train the model using the collected data
        pg_loss, v_loss, ent = agent.train(mb_states, mb_actions, mb_values, mb_advs, mb_returns, mb_old_a_logps)
        total_reward = mb_rewards.sum()
        mean_total_reward += total_reward
        mean_length += len(mb_states)
        writer.add_scalar("reward/total", total_reward, i)
        print("[Episode {:4d}] total reward = {:.6f}, length = {:d}".format(i, total_reward, len(mb_states)))

        #Save best model on improved return
        if total_reward > best_return:
            print("New best return! Saving the best model ... ")
            best_return = total_reward
            torch.save({
                "it": i,
                "best_return": best_return,
                "PolicyNet": policy_net.state_dict(),
                "ValueNet": value_net.state_dict()
            }, os.path.join(save_dir, "best_model.pt"))

        #Show the current result & save the model
        if i % 200 == 0:
            print("\n[{:5d} / {:5d}]".format(i, max_episode))
            print("----------------------------------")
            print("actor loss = {:.6f}".format(pg_loss))
            print("critic loss = {:.6f}".format(v_loss))
            print("entropy = {:.6f}".format(ent))
            print("mean return = {:.6f}".format(mean_total_reward / 200))
            print("mean length = {:.2f}".format(mean_length / 200))
            print("\nSaving the model ... ", end="")
            torch.save({
                "it": i,
                "PolicyNet": policy_net.state_dict(),
                "ValueNet": value_net.state_dict()
            }, os.path.join(save_dir, "model.pt"))
            print("Done.")
            print()
            # play(policy_net)
            mean_total_reward = 0
            mean_length = 0

    writer.close()

if __name__ == '__main__':
    #Create the environment
    max_episode_steps = 1200
    env = gym.make('BipedalWalker-v3',max_episode_steps=max_episode_steps)
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    print(s_dim)
    print(a_dim)

    #Create the policy net & value net
    policy_net = PolicyNet(s_dim, a_dim)
    value_net = ValueNet(s_dim)
    print(policy_net)
    print(value_net)

    #Create the runner (time_penalty: 每模拟秒扣除的分数)
    runner = EnvRunner(s_dim, a_dim, time_penalty=1.0, max_step=max_episode_steps)

    #Create a PPO agent for training
    agent = PPO(policy_net, value_net)

    #Train the network
    train(env, runner, policy_net, value_net, agent, max_episode=2000)
    env.close()
