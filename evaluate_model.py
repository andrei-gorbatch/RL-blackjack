import torch
import pandas as pd
import time
from datetime import datetime
import argparse
import os

# Import the necessary classes from the training script
from train_rl import BlackjackEnvironment, BlackjackDQNAgent

class BlackjackEvaluator:
    """Evaluator for trained blackjack DQN models."""
    
    def __init__(self, model_path: str):
        """Initialize evaluator with trained model."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        
        # Load the trained model
        self.agent = BlackjackDQNAgent()
        self.agent.q_network.load_state_dict(torch.load(model_path, map_location=self.device))
        self.agent.q_network.eval()
        
        # Set epsilon to 0 for evaluation (no exploration)
        self.agent.epsilon = 0.0
        
        self.env = BlackjackEnvironment()
        
    def evaluate_model(self, num_games: int = 10000, verbose: bool = True):
        """Evaluate the model over specified number of games."""
        if verbose:
            print(f"Evaluating model over {num_games} games...")
        
        # Statistics tracking
        stats = {
            'wins': 0,
            'losses': 0,
            'ties': 0,
            'total_reward': 0.0
        }
        
        start_time = time.time()
        
        for game_num in range(num_games):
            state = self.env.reset()
            done = False
            game_reward = 0.0
            
            while not done:
                action = self.agent.get_action(state, training=False)
                next_state, reward, done, info = self.env.step(action)
                state = next_state
                game_reward += reward
                
                if done:
                    stats['total_reward'] += game_reward
                    
                    # Categorize outcome
                    if game_reward > 0:
                        stats['wins'] += 1
                    elif game_reward < 0:
                        stats['losses'] += 1
                    else:
                        stats['ties'] += 1
                    
                    break
            
            # Progress update
            if verbose and (game_num + 1) % 1000 == 0:
                elapsed = time.time() - start_time
                games_per_sec = (game_num + 1) / elapsed
                print(f"  Completed {game_num + 1}/{num_games} games ({games_per_sec:.1f} games/sec)")
        
        evaluation_time = time.time() - start_time
        
        # Calculate final statistics
        total_games = stats['wins'] + stats['losses'] + stats['ties']
        win_rate = stats['wins'] / total_games if total_games > 0 else 0
        avg_reward = stats['total_reward'] / total_games if total_games > 0 else 0
        
        results = {
            'model_path': self.model_path,
            'evaluation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_games': total_games,
            'wins': stats['wins'],
            'losses': stats['losses'],
            'ties': stats['ties'],
            'win_rate': win_rate,
            'loss_rate': stats['losses'] / total_games if total_games > 0 else 0,
            'tie_rate': stats['ties'] / total_games if total_games > 0 else 0,
            'average_reward': avg_reward,
            'total_reward': stats['total_reward'],
            'evaluation_time_seconds': evaluation_time
        }
        
        if verbose:
            print(f"\nEvaluation completed in {evaluation_time:.2f} seconds")
            print("\nResults:")
            print(f"  Win Rate: {win_rate:.1%}")
            print(f"  Loss Rate: {results['loss_rate']:.1%}")
            print(f"  Tie Rate: {results['tie_rate']:.1%}")
            print(f"  Average Reward: {avg_reward:.3f}")
        
        return results
    
    def save_results_to_csv(self, results: dict, output_file: str):
        """Save evaluation results to CSV file."""      
        # Create DataFrame with results
        df = pd.DataFrame([results])
        
        # Check if file exists to determine if we need headers
        file_exists = os.path.exists(output_file)
        
        # Append to existing file or create new one
        if file_exists:
            df.to_csv(output_file, mode='a', header=False, index=False)
            print(f"\nResults appended to: {output_file}")
        else:
            df.to_csv(output_file, index=False)
            print(f"\nResults saved to new file: {output_file}")
        
        return output_file

def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate trained blackjack DQN model')
    parser.add_argument('--model', type=str, default='models/blackjack_dqn_model_20250707_162810.pth',
                        help='Path to trained model file')
    parser.add_argument('--games', type=int, default=10000,
                        help='Number of games to evaluate')
    parser.add_argument('--output', type=str, default="blackjack_evaluation.csv",
                        help='Output CSV file name')
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found!")
        return
    
    print("=== Blackjack Model Evaluation ===")
    print(f"Model: {args.model}")
    print(f"Games: {args.games}")
    print()
    
    # Initialize evaluator
    evaluator = BlackjackEvaluator(args.model)
    
    # Run evaluation
    results = evaluator.evaluate_model(args.games)
    
    # Save results
    evaluator.save_results_to_csv(results, args.output)

if __name__ == "__main__":
    main()