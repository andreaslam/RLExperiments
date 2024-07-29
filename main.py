import gymnasium as gym
import random
from tqdm import tqdm
import torch
import trainer
import buffer
import model
import torch.optim as optim

# configure gymnasium setup

# env = gym.make("ALE/Blackjack-v5", render_mode="human")
env = gym.make("CartPole-v1")

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

agent_net = torch.jit.script(
    model.AgentNet(observation_space, action_space, num_hidden_nodes=100, num_layers=5)
).to(device)

agent_trainer = trainer.Trainer(optim.AdamW(agent_net.parameters(), lr=1e-3))
agent_net.eval()
with torch.no_grad():
    torch.jit.save(agent_net, agent_trainer.path_name)

# training loop

TOTAL_TRAINING_STEPS = 100000000
GAMMA_DISCOUNT_FACTOR = 0.99
EPSILON_GREEDY_FACTOR = 0.9

REPLAY_BUFFER_SIZE = 4096
REPLAY_BUFFER_SAMPLE_SIZE = 2048


replay_buffer = buffer.ReplayBuffer(
    max_buffer_size=REPLAY_BUFFER_SIZE, batch_size=REPLAY_BUFFER_SAMPLE_SIZE
)

simulation = buffer.Simulation()

for _ in tqdm(range(TOTAL_TRAINING_STEPS), desc="training neural network"):
    nn_inputs = torch.tensor(observation, dtype=torch.float32).to(device).reshape(-1)
    nn_value_logits, nn_policy_logits = agent_net(nn_inputs.unsqueeze(0))
    nn_value, nn_q_head = (
        nn_value_logits.squeeze().cpu(),
        nn_policy_logits.squeeze().cpu(),
    )

    # add policy noise to increase exploration

    if random.random() > EPSILON_GREEDY_FACTOR:
        actions = (
            torch.tensor([random.uniform(-1.0, 1.0) for _ in range(len(nn_q_head))])
            * nn_q_head
        )

    else:
        actions = nn_q_head

    # env.render()

    position_data = env.step(actions.cpu().detach().numpy().argmax())

    position = buffer.Position(position_data, actions)

    simulation.append_position(position)

    observation, reward, terminated, truncated, info = position_data

    if terminated or truncated:
        observation, info = env.reset()
        replay_buffer.import_positions(simulation.export_positions())
        simulation = buffer.Simulation()

    if replay_buffer.get_buffer_capacity() >= replay_buffer.max_buffer_size:
        agent_net.train()
        agent_trainer.train_step(replay_buffer.sample_batch())
        agent_net.eval()
        with torch.no_grad():
            torch.jit.save(agent_net, agent_trainer.path_name)

env.close()
