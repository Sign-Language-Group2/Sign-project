
from game_class import Game

if __name__ == '__main__':
    model_path = './Game/model/model.p'  # Replace with the actual path to your model file
    game = Game(model_path)
    game.start_game()