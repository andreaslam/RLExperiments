import gymnasium as gym
import torch

env = gym.make("CartPole-v1", render_mode="human")

observation, info = env.reset()

# load neural networks
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

observation_space = len(observation.reshape(-1))
action_space = env.action_space.n.item()

agent_net = torch.jit.load("agent_nets/agent_gen_14.pt", device).eval()

returns = 0.0
total_returns = 0.0
episodes = 0

while True:
    nn_inputs = torch.tensor(observation, dtype=torch.float32).to(device).reshape(-1)
    nn_value_logits, nn_policy_logits = agent_net(nn_inputs.unsqueeze(0))
    nn_value, nn_q_head = (
        nn_value_logits.squeeze().cpu(),
        nn_policy_logits.squeeze().cpu(),
    )
    actions = nn_q_head.cpu().detach().numpy()

    # env.render()
    position_data = env.step(actions.argmax())
    observation, reward, terminated, truncated, info = position_data
    returns += reward
    total_returns += reward

    if terminated or truncated:
        env.reset()
        episodes += 1
        print(
            f"return (undiscounted): {returns}, average (undiscounted): {total_returns/episodes:.3f}"
        )
        returns = 0.0
