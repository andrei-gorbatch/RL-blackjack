# RL-blackjack
RL model that plays blackjack


## blackjack_game.py

### Features:
- **Human vs Dealer**: Interactive command-line gameplay
- **AI Player**: Automated gameplay using basic blackjack strategy

### Usage
#### Human Player Mode
```bash
python blackjack.py
```

#### Automated AI Mode
```bash
python blackjack.py auto
```


## blackjack_rl_basic.py

### Features:
- **Training**: Trains a RL (Deep Q-Network) to play blackjack, provided basic information: player total value, Ace, dealer visible card. Model saved to models/ folder.
- **Evaluation**: Automated evaluation run every N steps, graph saved in graphs/ folder. Runs evaluation at the end of the script as well.

### Usage
```bash
python blackjack_rl_basic.py
```

## blackjack_rl_extra_inputs.py

### Features:
- **Training**: Trains a RL (Deep Q-Network) to play blackjack, provided basic information as above, as well as information that can be used for card counting.
- **Evaluation**: Automated evaluation run every N steps, graph saved in graphs/ folder. Runs evaluation at the end of the script as well.

### Usage
```bash
python blackjack_rl_extra_inputs.py
```


## evaluate_model.py

### Features:
- **Evaluation**: Runs evaluation of a specified model. Saves output to blackjack_evaluation.csv. Allows easy comparison between models.

### Usage
```bash
python evaluate_model.py --model "path_to_model"
```