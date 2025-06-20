import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class DQNPricingAgent(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128, learning_rate=0.005):
        super(DQNPricingAgent, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)
        self.relu = nn.ReLU()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.memory = deque(maxlen=2000)
        self.batch_size = 512
        self.epsilon = 0.9
        self.epsilon_min = 0.05
        self.epsilon_decay = 400
        self.step_count = 0
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    def act(self, state):
        self.step_count += 1
        self.epsilon = self.epsilon_min + (0.9 - self.epsilon_min) * np.exp(-self.step_count / self.epsilon_decay)
        if np.random.random() <= self.epsilon:
            return random.randrange(len(self.get_action_space()))
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.forward(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())
    def replay(self, gamma=0.95):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        current_q_values = self.forward(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.forward(next_states).max(1)[0].detach()
        target_q_values = rewards + (gamma * next_q_values * ~dones)
        loss = self.loss_fn(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    def get_action_space(self):
        return np.arange(10, 501, 10)
class PricingEnvironment:
    def __init__(self, q0=5000, k=20, a=300, b=100, u=100, T=20):
        self.q0 = q0
        self.k = k
        self.a = a
        self.b = b
        self.u = u
        self.T = T
        self.reset()
    def reset(self):
        self.current_step = 0
        self.price_history = [100]
        self.total_profit = 0
        return self.get_state()
    def get_state(self):
        state = [self.current_step / self.T]
        prices = self.price_history[-5:] if len(self.price_history) >= 5 else [self.price_history[0]] * (5 - len(self.price_history)) + self.price_history
        state.extend([p / 500.0 for p in prices])
        return np.array(state)
    def demand_function(self, price_t, price_t_minus_1):
        price_increase = max(0, price_t - price_t_minus_1)
        price_decrease = max(0, price_t_minus_1 - price_t)
        demand = self.q0 - self.k * price_t - self.a * price_increase + self.b * price_decrease
        return max(0, demand)
    def profit_function(self, price_t, price_t_minus_1):
        demand = self.demand_function(price_t, price_t_minus_1)
        profit = demand * (price_t - self.u)
        return max(0, profit)
    def step(self, action):
        action_space = np.arange(10, 501, 10)
        price_t = action_space[action]
        price_t_minus_1 = self.price_history[-1] if self.price_history else 100
        reward = self.profit_function(price_t, price_t_minus_1)
        self.price_history.append(price_t)
        self.total_profit += reward
        self.current_step += 1
        done = self.current_step >= self.T
        next_state = self.get_state() if not done else None
        return next_state, reward, done

def train_pricing_agent(episodes=1000):
    env = PricingEnvironment()
    state_size = 6
    action_size = len(env.get_action_space())
    agent = DQNPricingAgent(state_size, action_size)
    scores = []
    losses = []
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        episode_losses = []
        while True:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            if not done:
                agent.remember(state, action, reward, next_state, done)
                state = next_state
            else:
                agent.remember(state, action, reward, np.zeros(state_size), done)
            total_reward += reward
            if len(agent.memory) > agent.batch_size:
                loss = agent.replay()
                if loss:
                    episode_losses.append(loss)
            if done:
                break
        scores.append(total_reward)
        if episode_losses:
            losses.append(np.mean(episode_losses))
        if episode % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print(f"Episode {episode}, Average Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.3f}")
    return agent, scores, losses
if __name__ == "__main__":
    trained_agent, training_scores, training_losses = train_pricing_agent(episodes=500)
    env = PricingEnvironment()
    state = env.reset()
    test_prices = []
    test_profits = []
    while True:
        action = trained_agent.act(state)
        action_space = np.arange(10, 501, 10)
        price = action_space[action]
        test_prices.append(price)
        next_state, reward, done = env.step(action)
        test_profits.append(reward)
        if done:
            break
        state = next_state
    print(f"Total profit: ${sum(test_profits):,.2f}")
    print(f"Average price: ${np.mean(test_prices):.2f}")
