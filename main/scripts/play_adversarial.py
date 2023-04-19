import torch
from transformers import pipeline
from transformers import AutoTokenizer
import chess.svg
import chess.pgn
import random
import string
import matplotlib.pyplot as plt

def load_model():
    torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.cuda.current_device()

    tokenizer = AutoTokenizer.from_pretrained("royal42/chess_tokenizer", use_fast=True)
    
    pipe = pipeline(
    "text-generation", model="royal42/test2", tokenizer=tokenizer, device=device
    )

    return pipe


def main():
    board = chess.Board()
    num_moves = 0
    model_1_incorrect = []
    model_2_incorrect = []

    model_1 = load_model()
    model_2 = load_model()

    current_sequence = ['e4']
    board.push_san('e4')
    while not board.is_game_over():
        # Model_1
        current_sequence, board, num_incorrect = make_move(model_1, current_sequence, board)
        model_1_incorrect.append(num_incorrect)
        num_moves += 1

        if board.is_game_over():
            break
        if num_moves >= 150:
            print('MAX MOVES REACHED')
            break

        # Model_2
        current_sequence, board, num_incorrect = make_move(model_2, current_sequence, board)
        model_2_incorrect.append(num_incorrect)
        num_moves += 1

        if num_moves >= 100:
            print('MAX MOVES REACHED')
            break

    print('Game over')
    print('Final board:')
    print(board)
    
    print('Outcome: ')
    print(chess.Board.outcome(board))
    #print('Winner: ')
    #print(chess.Board.outcome(board).winner())

    # Export to pgn
    game = chess.pgn.Game()
    game = game.from_board(board=board)

    name = ''.join(random.choices(string.ascii_uppercase +
                             string.digits, k=7))
    #print(game, file=open("./data/games/" + name + ".pgn", "w"), end="\n\n")

    # Run analysis
    #analyze(num_moves, model_1_incorrect, model_2_incorrect)

    return model_1_incorrect, model_2_incorrect


def make_move(model, current_sequence, board):
    num_incorrect = 0
    while True:
        move= gpt_move(model, current_sequence)
        print('Computer move: ', move)
        if is_illegal(board, move):
            print('Computer selected illegal move.')
            num_incorrect += 1
 
            if num_incorrect >= 15:
                move = board.san(random.choice(list(board.legal_moves)))
                print('Selected random move: ', move)
            else:
                continue
        current_sequence.append(move)
        board.push_san(move)
        print(board)
        break

    return current_sequence, board, num_incorrect

def gpt_move(model, current_sequence):
    new_sequence = list(current_sequence)
    string_sequence= ' '.join(new_sequence)
    output = model(string_sequence, num_return_sequences=1, max_new_tokens=6, pad_token_id=model.tokenizer.eos_token_id)[0]["generated_text"].split(' ')
    new_move = output[len(new_sequence)]
    return new_move

def is_illegal(board, move):
    try:
        if board.parse_san(move) in board.legal_moves:
            return False
        else:
            return True
    except:
        return True
    
def analyze(num_moves, model_1_incorrect, model_2_incorrect):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(num_moves//2), model_1_incorrect[:num_moves//2], label="model_1")
    ax.plot(range(num_moves//2), model_2_incorrect[:num_moves//2], label="model_2")
    plt.legend()
    plt.show()

# For running multiple games for analysis
def main_wrapper():
    model_1_incorrect_games = []
    model_2_incorrect_games = []

    for _ in range(10):
        m1, m2 = main()
        model_1_incorrect_games.append(m1)
        model_2_incorrect_games.append(m2)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for game in model_1_incorrect_games:
        ax.plot(range(len(game)), game)
    for game in model_2_incorrect_games:
        ax.plot(range(len(game)), game)
    plt.show()


if __name__ == "__main__":
    main_wrapper()
