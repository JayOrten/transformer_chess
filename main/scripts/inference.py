import torch
from transformers import pipeline

def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    pipe = pipeline(
        "text-generation", model="royal42/test2", device=device
    )

    input = ""

    print(pipe(input, num_return_sequences=1))[0]["generated_text"]

if __name__ == "__main__":
    main()