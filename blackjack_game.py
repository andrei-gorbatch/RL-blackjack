import random
import logging

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('blackjack_game.log')
    ]
)
logger = logging.getLogger(__name__)

class Card:
    """Represents a playing card."""
    
    def __init__(self, suit: str, rank: str):
        self.suit = suit
        self.rank = rank
    
    def __str__(self):
        return f"{self.rank} of {self.suit}"
    
    def get_value(self) -> int:
        """Returns the blackjack value of the card."""
        if self.rank in ['Jack', 'Queen', 'King']:
            return 10
        elif self.rank == 'Ace':
            return 11  # Ace handling is done in Hand class
        else:
            return int(self.rank)

class Deck:
    """Represents a deck of 52 playing cards."""
    
    def __init__(self):
        self.cards = []
        self.reset_deck()
    
    def reset_deck(self):
        """Creates a fresh deck of 52 cards."""
        suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King', 'Ace']
        
        self.cards = [Card(suit, rank) for suit in suits for rank in ranks]
        self.shuffle()
    
    def shuffle(self):
        """Shuffles the deck."""
        random.shuffle(self.cards)
    
    def deal_card(self) -> Card:
        """Deals one card from the deck."""
        if len(self.cards) == 0:
            self.reset_deck()
        return self.cards.pop()

class Hand:
    """Represents a hand of cards."""
    
    def __init__(self):
        self.cards = []
    
    def add_card(self, card: Card):
        """Adds a card to the hand."""
        self.cards.append(card)
    
    def get_value(self) -> int:
        """Calculates the best possible value of the hand."""
        value = 0
        aces = 0
        
        for card in self.cards:
            if card.rank == 'Ace':
                aces += 1
                value += 11
            else:
                value += card.get_value()
        
        # Adjust for aces
        while value > 21 and aces > 0:
            value -= 10
            aces -= 1
        
        return value
    
    def is_bust(self) -> bool:
        """Checks if the hand is bust (over 21)."""
        return self.get_value() > 21
    
    def is_blackjack(self) -> bool:
        """Checks if the hand is a blackjack (21 with 2 cards)."""
        return len(self.cards) == 2 and self.get_value() == 21
    
    def __str__(self):
        return ', '.join(str(card) for card in self.cards)

class AutomatedPlayer:
    """AI player that makes decisions based on basic blackjack strategy."""
    
    def __init__(self):
        self.name = "AI Player"
    
    def make_decision(self, player_hand: Hand, dealer_visible_card: Card) -> str:
        """
        Makes a decision based on basic blackjack strategy.
        Returns 'hit' or 'stand'.
        """
        player_value = player_hand.get_value()
        dealer_value = dealer_visible_card.get_value()
        
        # Basic strategy rules
        if player_value <= 11:
            return 'hit'  # Always hit on 11 or less
        elif player_value >= 17:
            return 'stand'  # Always stand on 17 or more
        elif player_value == 16:
            return 'hit' if dealer_value >= 7 else 'stand'
        elif player_value == 15:
            return 'hit' if dealer_value >= 7 else 'stand'
        elif player_value == 14:
            return 'hit' if dealer_value >= 7 else 'stand'
        elif player_value == 13:
            return 'hit' if dealer_value >= 7 else 'stand'
        elif player_value == 12:
            return 'hit' if dealer_value >= 7 or dealer_value <= 3 else 'stand'
        else:
            return 'hit'  # Default to hit

class BlackjackGame:
    """Main game class that handles the blackjack game logic."""
    
    def __init__(self, automated: bool = False):
        self.deck = Deck()
        self.player_hand = Hand()
        self.dealer_hand = Hand()
        self.other_players_hand = Hand()
        self.game_over = False
        self.automated = automated
        self.ai_player = AutomatedPlayer() if automated else None
    
    def start_new_game(self):
        """Starts a new game of blackjack."""
        self.deck.reset_deck()  # Reset and shuffle deck for each new game
        self.player_hand = Hand()
        self.dealer_hand = Hand()
        self.game_over = False
        
        # Deal initial cards
        self.player_hand.add_card(self.deck.deal_card())
        self.dealer_hand.add_card(self.deck.deal_card())
        self.player_hand.add_card(self.deck.deal_card())
        self.dealer_hand.add_card(self.deck.deal_card())
        
        logger.info("=" * 50)
        logger.info("NEW BLACKJACK GAME")
        logger.info("=" * 50)
        self.display_hands(hide_dealer_card=True)
        
        # Check for blackjacks
        if self.player_hand.is_blackjack() and self.dealer_hand.is_blackjack():
            logger.info("Both have blackjack! It's a tie!")
            self.game_over = True
            return "tie"
        elif self.player_hand.is_blackjack():
            logger.info("Blackjack! You win!")
            self.game_over = True
            return "player_blackjack"
        elif self.dealer_hand.is_blackjack():
            logger.info("Dealer has blackjack! You lose!")
            self.display_hands(hide_dealer_card=False)
            self.game_over = True
            return "dealer_blackjack"
        
        return "continue"
    
    def display_hands(self, hide_dealer_card: bool = False):
        """Displays the current hands."""
        logger.info(f"Your hand: {self.player_hand} (Value: {self.player_hand.get_value()})")
        
        if hide_dealer_card and len(self.dealer_hand.cards) > 0:
            logger.info(f"Dealer's hand: {self.dealer_hand.cards[0]}, [Hidden Card]")
        else:
            logger.info(f"Dealer's hand: {self.dealer_hand} (Value: {self.dealer_hand.get_value()})")
    
    def player_turn(self) -> str:
        """Handles the player's turn."""
        while not self.game_over:
            if self.automated:
                # AI makes decision
                dealer_visible_card = self.dealer_hand.cards[0]
                decision = self.ai_player.make_decision(self.player_hand, dealer_visible_card)
                logger.info(f"AI Decision: {decision.upper()}")          
                choice = decision
            else:
                # Human player input
                logger.info("Your turn:")
                logger.info("1. Hit (h)")
                logger.info("2. Stand (s)")
                choice = input("Enter your choice (h/s): ").lower().strip()
            
            if choice in ['h', 'hit']:
                self.player_hand.add_card(self.deck.deal_card())
                self.display_hands(hide_dealer_card=True)
                
                if self.player_hand.is_bust():
                    logger.info("Bust! You lose!")
                    self.game_over = True
                    return "player_bust"
                
            elif choice in ['s', 'stand']:
                logger.info("You stand." if not self.automated else "AI stands.")
                break
            else:
                if not self.automated:
                    logger.warning("Invalid choice. Please enter 'h' for hit or 's' for stand.")
        
        return "continue"
    
    def dealer_turn(self) -> str:
        """Handles the dealer's turn."""
        logger.info("Dealer's turn:")
        self.display_hands(hide_dealer_card=False)
        
        while self.dealer_hand.get_value() < 17:
            logger.info("Dealer hits...")
            self.dealer_hand.add_card(self.deck.deal_card())
            self.display_hands(hide_dealer_card=False)
            
            if self.dealer_hand.is_bust():
                logger.info("Dealer busts! You win!")
                return "dealer_bust"
        
        logger.info("Dealer stands.")
        return "continue"
    
    def determine_winner(self) -> str:
        """Determines the winner of the game."""
        player_value = self.player_hand.get_value()
        dealer_value = self.dealer_hand.get_value()
        
        logger.info("Final Results:")
        if self.automated:
            logger.info(f"AI hand: {player_value}")
        else:
            logger.info(f"Your hand: {player_value}")
        logger.info(f"Dealer's hand: {dealer_value}")
        
        if player_value > dealer_value:
            if self.automated:
                logger.info("AI wins!")
            else:
                logger.info("You win!")
            return "player_win"
        elif dealer_value > player_value:
            logger.info("Dealer wins!")
            return "dealer_win"
        else:
            logger.info("It's a tie!")
            return "tie"
    
    def play_game(self) -> float:
        """Main game loop. Returns game result as float for automated games."""
        if self.automated:
            logger.info("Welcome to Automated Blackjack!")
            logger.info("AI will play one game using basic blackjack strategy.")
        else:
            logger.info("Welcome to Blackjack!")
            logger.info("Goal: Get as close to 21 as possible without going over.")
            logger.info("Face cards are worth 10, Aces are worth 1 or 11.")
        
        # Play single game
        result = self.start_new_game()
        game_result = None
        
        if result == "continue":
            player_result = self.player_turn()
            
            if player_result == "continue":
                dealer_result = self.dealer_turn()
                
                if dealer_result == "continue":
                    game_result = self.determine_winner()
                elif dealer_result == "dealer_bust":
                    game_result = "player_win"
            elif player_result == "player_bust":
                game_result = "dealer_win"
        else:
            # Handle blackjack cases
            if result == "player_blackjack":
                game_result = "player_win"
            elif result == "dealer_blackjack":
                game_result = "dealer_win"
            elif result == "tie":
                game_result = "tie"
        
        # Convert result to float for automated games
        if self.automated:
            if game_result == "player_win":
                result_float = 1.0
                logger.info(f"Game Result: {result_float} (AI Win)")
            elif game_result == "dealer_win":
                result_float = 0.0
                logger.info(f"Game Result: {result_float} (Dealer Win)")
            else:  # tie
                result_float = 0.5
                logger.info(f"Game Result: {result_float} (Tie)")
            
            logger.info("Automated game complete!")
            return result_float
        else:
            # Ask if player wants to play again (only for human players)
            while True:
                play_again = input("Would you like to play again? (y/n): ").lower().strip()
                if play_again in ['y', 'yes']:
                    # Restart the game loop
                    return self.play_game()
                elif play_again in ['n', 'no']:
                    logger.info("Thanks for playing!")
                    return None
                else:
                    logger.warning("Please enter 'y' for yes or 'n' for no.")

def main():
    """Main function to run the blackjack game."""
    import sys
    
    # Check if automated mode is requested
    if len(sys.argv) > 1 and sys.argv[1].lower() in ['auto', 'automated', 'ai']:
        game = BlackjackGame(automated=True)
    else:
        game = BlackjackGame(automated=False)
    
    game.play_game()

def run_automated_game() -> float:
    """Function to run a single automated game - useful for scripting."""
    game = BlackjackGame(automated=True)
    result = game.play_game()
    return result  # Returns float: 1.0 for AI win, 0.0 for dealer win, 0.5 for tie

if __name__ == "__main__":
    main()