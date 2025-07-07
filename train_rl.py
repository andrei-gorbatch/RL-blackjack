import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import logging
from datetime import datetime

# Import our blackjack game
from blackjack_game import BlackjackGame, Hand, Deck

# Set up logging for RL training
logging.basicConfig(level=logging.WARNING)  # Reduce blackjack game logging during training

class BlackjackEnvironment:
    """Environment wrapper for the blackjack game to work with RL agent."""
    
    def __init__(self):
        self.game = BlackjackGame(automated=False)
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset the environment and return initial state."""
        self.game = BlackjackGame(automated=False)
        self.game.deck = Deck()
        self.game.player_hand = Hand()
        self.game.dealer_hand = Hand()
        self.game.game_over = False
        
        # Deal initial cards
        self.game.player_hand.add_card(self.game.deck.deal_card())
        self.game.dealer_hand.add_card(self.game.deck.deal_card())
        self.game.player_hand.add_card(self.game.deck.deal_card())
        self.game.dealer_hand.add_card(self.game.deck.deal_card())
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Convert game state to numerical representation."""
        player_value = self.game.player_hand.get_value()
        dealer_visible = self.game.dealer_hand.cards[0].get_value()
        
        # Check for usable ace in player hand
        usable_ace = any(card.rank == 'Ace' for card in self.game.player_hand.cards) and player_value <= 21
        
        # State: [player_value, dealer_visible, usable_ace, is_blackjack]
        is_blackjack = self.game.player_hand.is_blackjack()
        
        return np.array([player_value, dealer_visible, int(usable_ace), int(is_blackjack)], dtype=np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action and return next state, reward, done, info."""
        reward = 0.0
        done = False
        info = {}
        
        if self.game.game_over:
            return self._get_state(), reward, True, info
        
        # Check for initial blackjack
        if self.game.player_hand.is_blackjack():
            done = True
            if self.game.dealer_hand.is_blackjack():
                reward = 0.0  # Tie
                info['result'] = 'tie'
            else:
                reward = 1.5  # Blackjack bonus
                info['result'] = 'blackjack'
            return self._get_state(), reward, done, info
        
        if action == 0:  # Stand
            # Dealer plays
            while self.game.dealer_hand.get_value() < 17:
                self.game.dealer_hand.add_card(self.game.deck.deal_card())
            
            done = True
            player_value = self.game.player_hand.get_value()
            dealer_value = self.game.dealer_hand.get_value()
            
            if dealer_value > 21:  # Dealer busts
                reward = 1.0
                info['result'] = 'dealer_bust'
            elif player_value > dealer_value:
                reward = 1.0
                info['result'] = 'player_win'
            elif player_value < dealer_value:
                reward = -1.0
                info['result'] = 'dealer_win'
            else:
                reward = 0.0
                info['result'] = 'tie'
                
        elif action == 1:  # Hit
            self.game.player_hand.add_card(self.game.deck.deal_card())
            
            if self.game.player_hand.is_bust():
                reward = -1.0
                done = True
                info['result'] = 'player_bust'
            else:
                # Continue playing
                reward = 0.0
                done = False
        
        self.game.game_over = done
        return self._get_state(), reward, done, info

class DQN(nn.Module):
    """Deep Q-Network for blackjack."""
    
    def __init__(self, input_size: int = 4, hidden_size: int = 128, output_size: int = 2):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class ReplayBuffer:
    """Experience replay buffer for DQN."""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
        self.experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        exp = self.experience(state, action, reward, next_state, done)
        self.buffer.append(exp)
    
    def sample(self, batch_size: int):
        """Sample batch of experiences."""
        experiences = random.sample(self.buffer, batch_size)
        
        states = torch.FloatTensor([e.state for e in experiences])
        actions = torch.LongTensor([e.action for e in experiences])
        rewards = torch.FloatTensor([e.reward for e in experiences])
        next_states = torch.FloatTensor([e.next_state for e in experiences])
        dones = torch.BoolTensor([e.done for e in experiences])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class BlackjackDQNAgent:
    """DQN Agent for learning blackjack."""
    
    def __init__(self, state_size: int = 4, action_size: int = 2, lr: float = 0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        
        # Neural networks
        self.q_network = DQN(state_size, 128, action_size).to(self.device)
        self.target_network = DQN(state_size, 128, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Replay buffer
        self.memory = ReplayBuffer(10000)
        
        # Hyperparameters
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.update_target_every = 100
        
        # Training stats
        self.losses = []
        self.rewards = []
        self.win_rates = []
        
    def get_action(self, state: np.ndarray, training: bool = True) -> int:
        """Choose action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.push(state, action, reward, next_state, done)
    
    def replay(self):
        """Train the agent on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Target Q values
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.losses.append(loss.item())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        """Update target network with current network weights."""
        self.target_network.load_state_dict(self.q_network.state_dict())

def train_agent(episodes: int = 10000, eval_every: int = 500):
    """Train the DQN agent to play blackjack."""
    env = BlackjackEnvironment()
    agent = BlackjackDQNAgent()
    
    print(f"Training on {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    print(f"Starting training for {episodes} episodes...")
    
    episode_rewards = []
    recent_rewards = deque(maxlen=100)
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        # Train the agent
        agent.replay()
        
        # Update target network
        if episode % agent.update_target_every == 0:
            agent.update_target_network()
        
        episode_rewards.append(total_reward)
        recent_rewards.append(total_reward)
        
        # Evaluation and logging
        if episode % eval_every == 0:
            avg_reward = np.mean(recent_rewards) if recent_rewards else 0
            win_rate = evaluate_agent(agent, env, 1000)
            agent.win_rates.append(win_rate)
            
            print(f"Episode {episode}")
            print(f"  Average Reward (last 100): {avg_reward:.3f}")
            print(f"  Win Rate: {win_rate:.1%}")
            print(f"  Epsilon: {agent.epsilon:.3f}")
            print(f"  Replay Buffer Size: {len(agent.memory)}")
            print()
    
    print("Training completed!")
    return agent, episode_rewards

def evaluate_agent(agent: BlackjackDQNAgent, env: BlackjackEnvironment, num_games: int = 1000) -> float:
    """Evaluate the agent's performance."""
    wins = 0
    total_games = 0
    
    for _ in range(num_games):
        state = env.reset()
        done = False
        
        while not done:
            action = agent.get_action(state, training=False)
            next_state, reward, done, info = env.step(action)
            state = next_state
            
            if done:
                total_games += 1
                if reward > 0:
                    wins += 1
                break
    
    return wins / total_games if total_games > 0 else 0

def plot_training_results(agent: BlackjackDQNAgent, episode_rewards: List[float], timestamp: str):
    """Plot training results."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Episode rewards
    ax1.plot(episode_rewards)
    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.grid(True)
    
    # Moving average of rewards
    window_size = 100
    if len(episode_rewards) >= window_size:
        moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
        ax2.plot(moving_avg)
        ax2.set_title(f'Moving Average Reward (window={window_size})')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Average Reward')
        ax2.grid(True)
    
    # Training loss
    if agent.losses:
        ax3.plot(agent.losses)
        ax3.set_title('Training Loss')
        ax3.set_xlabel('Training Step')
        ax3.set_ylabel('Loss')
        ax3.grid(True)
    
    # Win rate over time
    if agent.win_rates:
        ax4.plot(agent.win_rates)
        ax4.set_title('Win Rate During Training')
        ax4.set_xlabel('Evaluation Period')
        ax4.set_ylabel('Win Rate')
        ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"graphs/training_results_{timestamp}.png") 

def test_trained_agent(agent: BlackjackDQNAgent, num_games: int = 1000):
    """Test the trained agent and compare to basic strategy."""
    env = BlackjackEnvironment()
    
    print(f"Testing trained agent over {num_games} games...")
    
    # Test RL agent
    rl_wins = 0
    rl_games = 0
    
    for _ in range(num_games):
        state = env.reset()
        done = False
        
        while not done:
            action = agent.get_action(state, training=False)
            next_state, reward, done, info = env.step(action)
            state = next_state
            
            if done:
                rl_games += 1
                if reward > 0:
                    rl_wins += 1
                break
    
    rl_win_rate = rl_wins / rl_games if rl_games > 0 else 0
    
    print("RL Agent Results:")
    print(f"  Win Rate: {rl_win_rate:.1%}")
    print(f"  Games Won: {rl_wins}/{rl_games}")
    
    return rl_win_rate

def main():
    """Main training and evaluation loop."""
    print("=== Blackjack Reinforcement Learning Training ===")
    print()
    
    # Train the agent
    agent, episode_rewards = train_agent(episodes=10000, eval_every=200)
    
    # Plot results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_training_results(agent, episode_rewards, timestamp)
    
    # Final evaluation
    final_win_rate = test_trained_agent(agent, num_games=5000)
    
    # Save the trained model
    torch.save(agent.q_network.state_dict(), f'models/blackjack_dqn_model_{timestamp}.pth')
    print("\nModel saved as 'blackjack_dqn_model.pth'")
    
    print("\nFinal Results:")
    print(f"Final Win Rate: {final_win_rate:.1%}")
    print(f"Training Episodes: {len(episode_rewards)}")
    print(f"Final Epsilon: {agent.epsilon:.3f}")

if __name__ == "__main__":
    main()