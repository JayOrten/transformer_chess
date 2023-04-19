import torch
from transformers import pipeline
from transformers import AutoTokenizer
import chess.svg

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

    model = load_model()

    current_sequence = []
    while not board.is_game_over():
        # Your move
        while True:
            print('Your move: ')
            move = input()

            if is_illegal(board, move):
                print('That move is not legal, try again.')
                continue
            current_sequence.append(move)
            board.push_san(move)
            print(board)
            break

        if board.is_game_over():
            break

        # Computer move
        while True:
            move= gpt_move(model, current_sequence)
            print('Computer move: ', move)
            if is_illegal(board, move):
                print('Computer selected illegal move.')
                continue
            current_sequence.append(move)
            board.push_san(move)
            print(board)
            break

    print('Game over')

def gpt_move(model, current_sequence):
    new_sequence = list(current_sequence)
    string_sequence= ' '.join(new_sequence)
    output = model(string_sequence, num_return_sequences=1, max_new_tokens=4, pad_token_id=model.tokenizer.eos_token_id)[0]["generated_text"].split(' ')
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

if __name__ == "__main__":
    main()
