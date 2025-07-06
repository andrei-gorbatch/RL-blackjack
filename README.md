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

#### Programmatic Usage
```python
from blackjack import run_automated_game

# Run single automated game
result = run_automated_game()
print(f"Game result: {result}")
# Returns: 1.0 (AI win), 0.0 (dealer win), 0.5 (tie)
```