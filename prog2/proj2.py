#!pip install torch
import sentencepiece as spm
import os
import pickle

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import nltk
from nltk.translate.bleu_score import sentence_bleu
import math
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
import numpy as np
import argparse

def create_tokenizer(input_dir :str, output_path :str):
    """Creates a tokenizer using texts from input_dir and saves it to tokenizer.pkl"""
    
    model_prefix = "bpe_tokenizer"
    vocab_size = 10000  

    input_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.txt')]

    input_files_str = ",".join(input_files)

    
   
    spm.SentencePieceTrainer.Train(
        input=input_files_str,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe", 
        user_defined_symbols="<pad>"
    )

   
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(f"{model_prefix}.model")

    
    with open(output_path, "wb") as f:
        pickle.dump(tokenizer, f)
    return tokenizer

def make_jsonl_from_txt_dir(input_dir, tokenizer_path, output_dir, 
                                       min_len=2, max_len=250, test_ratio=0.2, 
                                       seed=42):
    # Load trained tokenizer
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    examples = []

    for file_name in os.listdir(input_dir):
        if file_name.endswith(".txt"):
            with open(os.path.join(input_dir, file_name), "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    # Tokenize the line
                    token_ids = tokenizer.encode(line, out_type=int)

                    if len(token_ids) < min_len + 1:
                        continue
                    if len(token_ids) > max_len:
                        token_ids = token_ids[:max_len]

                    # Random split point for next-token prediction
                    split_point = random.randint(min_len, len(token_ids) - 1)
                    for split_point in range(min_len, len(token_ids) - 2, 1):
                        
                        prompt_ids = token_ids[:split_point]
                        completion_ids = token_ids[split_point:split_point + 1]

                        prompt_text = tokenizer.decode(prompt_ids)
                        completion_text = tokenizer.decode(completion_ids)
                        if completion_text.strip() == "":
                            continue

                        examples.append({
                            "prompt": prompt_text,
                            "completion": completion_text
                        })

    # Split into train/test
    train_data, test_data = train_test_split(
        examples, test_size=test_ratio, random_state=seed
    )

    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train.jsonl")
    test_path = os.path.join(output_dir, "test.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for ex in train_data:
            json.dump(ex, f)
            f.write("\n")

    with open(test_path, "w", encoding="utf-8") as f:
        for ex in test_data:
            json.dump(ex, f)
            f.write("\n")

    print(f"Saved {len(train_data)} training examples to {train_path}")
    print(f"Saved {len(test_data)} testing examples to {test_path}")

class Prog2Model(nn.Module):
    def __init__(self, model_option: str, num_layers: int, tokenizer: spm.SentencePieceProcessor):
        super(Prog2Model, self).__init__()
        self.model_option = model_option
        self.num_layers = num_layers
        self.vocab_size = 10000
        self.tokenizer = tokenizer

        self.dropout = 0.4
        
        model_dim = 256
        
        self.positional_encoding = PositionalEncoding(d_model = model_dim)

        # Define the embedding layer
        self.embedding_layer = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=model_dim)

        # Define the model based on the selected option
        if self.model_option == "RNN":
            self.model = nn.RNN(input_size=model_dim, hidden_size=model_dim, num_layers=self.num_layers, batch_first=True,
                                dropout=self.dropout)
        elif self.model_option == "LSTM":
            self.model = nn.LSTM(input_size=model_dim, hidden_size=model_dim, num_layers=self.num_layers, batch_first=True,
                                 dropout=self.dropout)
        elif self.model_option == "Transformer":
            self.model = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=model_dim, nhead=8, dropout=self.dropout), num_layers=self.num_layers
            )
        else:
            raise ValueError("Invalid model option. Choose from 'RNN', 'LSTM', or 'Transformer'.")

        # Define the fully connected layer for output
        self.fc = nn.Linear(model_dim, self.vocab_size)

    def forward(self, tokens, temperature=1.0, train_mode=True):
        # Embed the tokens
        
        
        embedded_tokens = self.embedding_layer(tokens)
        
        if self.model_option == "Transformer":
            embedded_tokens = self.positional_encoding(embedded_tokens)

        embedded_tokens = nn.functional.dropout(embedded_tokens, p=self.dropout, training=train_mode)

        # Pass through the selected model
        if self.model_option in ["RNN", "LSTM"]:
            pre_fc_outputs, _ = self.model(embedded_tokens)
        elif self.model_option == "Transformer":
            pre_fc_outputs = self.model(embedded_tokens)

        # Compute the output probabilities for the next token
        logits = self.fc(pre_fc_outputs)
        if train_mode:
            return logits
        probabilities = nn.functional.softmax(logits / temperature, dim=-1)
        predicted_token = torch.argmax(probabilities, dim=-1)
        return predicted_token
    
    def prompt(self, prompt_text: str, max_seq_len: int = 50, temperature: float = 1.0):
        """Tokenizes the input text and returns the token IDs."""
        tokens = self.tokenizer.encode(prompt_text, out_type=int)
        while len(tokens) < max_seq_len: #also add eos
            tokens_tensor = torch.tensor([tokens], dtype=torch.long).to(next(self.parameters()).device)
            next_token = self(tokens_tensor, temperature, False).squeeze(0)[-1]
            tokens.append(next_token.item())
            if "<eos>" in self.tokenizer.IdToPiece(next_token.item()):
                break
        return tokens
    
        
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=250):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# Custom Dataset for JSONL data
class TextCompletionDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_seq_len=250):
        self.data = []
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        

        # Load and tokenize the data
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                prompt = tokenizer.encode(item["prompt"], out_type=int)
                completion = tokenizer.encode(item["completion"], out_type=int)
                self.data.append((prompt, completion))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt, completion = self.data[idx]

        # Truncate or pad the sequences to max_seq_len
        input_ids = prompt[:self.max_seq_len]
        target_ids = completion[:self.max_seq_len]

        # Pad sequences to max_seq_len
        input_ids += [0] * (self.max_seq_len - len(input_ids))
        target_ids += [0] * (self.max_seq_len - len(target_ids))

        return torch.tensor(input_ids), torch.tensor(target_ids)



# Training function with early stopping and learning rate scheduler
def train_model(model, train_dataloader, val_dataloader, optimizer, criterion, device, epochs=30, patience=5):
    model.train()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    best_val_loss = float('inf')
    patience_counter = 0

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        print("Starting epoch ", epoch)
        total_loss = 0
        model.train()

        inc = 1000
        # Training loop
        for i, batch in enumerate(train_dataloader):
            input_ids, target_ids = batch
            # print(batch)
            input_ids, target_ids = input_ids.to(device), target_ids.to(device)

            # Forward pass
            optimizer.zero_grad()
            # Find the actual length of each prompt (before padding)
            lengths = (input_ids != 0).sum(dim=1)  # [batch_size]

            # Forward pass
            outputs = model(input_ids)  # [batch_size, seq_len, vocab_size]

            # Gather the logits at the last prompt token position
            batch_indices = torch.arange(input_ids.size(0), device=device)
            last_token_logits = outputs[batch_indices, lengths - 1, :]  # [batch_size, vocab_size]

            # Use the first token of the target (single-token completion)
            target_ids = target_ids[:, 0]  # [batch_size]

            loss = criterion(last_token_logits, target_ids)
            total_loss += loss

            # Backward pass and optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) #handle exploding gradient
            optimizer.step()
            
            if i % inc == 0:
                print(f"Finished batch {i} / {len(train_dataloader)}")
            
        
        avg_train_loss = total_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)

        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids, target_ids = batch
                input_ids, target_ids = input_ids.to(device), target_ids.to(device)

                lengths = (input_ids != 0).sum(dim=1)  # [batch_size]

                # Forward pass
                outputs = model(input_ids)  # [batch_size, seq_len, vocab_size]

                # Gather the logits at the last prompt token position
                batch_indices = torch.arange(input_ids.size(0), device=device)
                last_token_logits = outputs[batch_indices, lengths - 1, :]  # [batch_size, vocab_size]

                # Use the first token of the target (single-token completion)
                target_ids = target_ids[:, 0]  # [batch_size]

                loss = criterion(last_token_logits, target_ids)
                
                val_loss += loss.item()

        val_loss /= len(val_dataloader)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {total_loss / len(train_dataloader):.4f}, Validation Loss: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"prog2_model_{model.model_option}.pth")
            print("Model improved and saved.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
    
    # Plot the training and validation loss curves
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(np.array(train_losses), label="Training Loss")
        plt.plot(np.array(val_losses), label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss Curves")
        plt.legend()
        plt.grid()
        plt.savefig(f"loss_curves_{model.model_option}.png")  # Save the plot as an image
        plt.show()
    except Exception as e:
        print(f"failed to plot loss: {e}")

def compute_bleu_score(model, dataloader, tokenizer, device):
    model.eval()
    bleu_scores = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids, target_ids = batch
            input_ids, target_ids = input_ids.to(device), target_ids.to(device)

            # Generate predictions
            predictions = []
            for i in range(input_ids.size(0)):
                prompt = tokenizer.decode(input_ids[i].cpu().tolist())
                predicted_tokens = model.prompt(prompt, max_seq_len=50)
                predicted_text = tokenizer.decode(predicted_tokens)
                predictions.append(predicted_text)

            # Compute BLEU score for each prediction
            for i in range(len(predictions)):
                reference = tokenizer.decode(target_ids[i].cpu().tolist()).split()
                candidate = predictions[i].split()
                bleu = sentence_bleu([reference], candidate)
                bleu_scores.append(bleu)

    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    print(f"Average BLEU Score: {avg_bleu:.4f}")
    return avg_bleu

# Function to compute perplexity
def compute_perplexity(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids, target_ids = batch
            input_ids, target_ids = input_ids.to(device), target_ids.to(device)

            # Forward pass
            outputs = model(input_ids)
            loss = criterion(outputs.view(-1, model.vocab_size), target_ids.view(-1))

            # Accumulate loss and token count
            total_loss += loss.item() * target_ids.numel()
            total_tokens += target_ids.numel()

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    print(f"Perplexity: {perplexity:.4f}")
    return perplexity

# Main script
if __name__ == "__main__":
    print("Starting...")

    #create tokenizer
    
    # Load the tokenizer
    # Parse script arguments
    parser = argparse.ArgumentParser(description="Train and evaluate a text completion model.")
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer.pkl", help="COMPLETE Path to the tokenizer pickle file (to be created).")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing the raw text data.")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save processed data and models.")
    parser.add_argument("--model_option", type=str, choices=["RNN", "LSTM", "Transformer"], default="LSTM", help="Model type to use.")
    parser.add_argument("--num_layers", type=int, default=8, help="Number of layers in the model.")
    parser.add_argument("--max_seq_length", type=int, default=250, help="Maximum sequence length.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training and evaluation.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate for the optimizer.")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for the optimizer.")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping.")
    parser.add_argument("--test_only", type=bool, default=True, help="Run evaluation only without training.")
    parser.add_argument("--recreate_data_and_tokenizer", action="store_true", help="Recreate the tokenizer and data files.")
    parser.add_argument("--prompt", type=str, default="Which do you prefer? Cats or Dogs?", help="The input prompt for inference.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature to sample from.")

    args = parser.parse_args()

    # Assign parsed arguments to variables
    tokenizer_path = args.tokenizer_path
    data_dir = args.data_dir
    output_dir = args.output_dir
    model_option = args.model_option #RNN, LSTM, Transformer
    num_layers = args.num_layers
    max_seq_length = args.max_seq_length
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    patience = args.patience
    test_only = args.test_only
    recreate_data_and_tokenizer = args.recreate_data_and_tokenizer
    inference_prompt = args.prompt
    temperature = args.temperature

    # tokenizer_path = "/home/gmuslow/prog2/Foundational-AI/tokenizer.pkl"
    # all_data_path = "/home/gmuslow/prog2/"
    train_path = f"{data_dir}/train.jsonl"
    test_path = f"{data_dir}/test.jsonl"
    
    if recreate_data_and_tokenizer:
        create_tokenizer(data_dir, tokenizer_path)
        make_jsonl_from_txt_dir(data_dir, tokenizer_path, data_dir)

    
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    

    
    divider = 3

    if not test_only:
        # Load the dataset
        train_dataset = TextCompletionDataset(
            file_path=train_path,
            tokenizer=tokenizer,
            max_seq_len=max_seq_length
        )
        
        subset_size = len(train_dataset) // divider
        train_dataset = torch.utils.data.Subset(train_dataset, range(subset_size))
        
        # Split the dataset into training and validation sets (80-20 split)
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

        # Create DataLoaders for training and validation sets
        train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        
        # Initialize the model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        model = Prog2Model(model_option=model_option, num_layers=num_layers, tokenizer=tokenizer).to(device)

        # Define the optimizer and loss function
        optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding index

        # Train the model
        train_model(model, train_dataloader, val_dataloader, optimizer, criterion, device, epochs=30)

        # Save the trained model
        torch.save(model.state_dict(), f"prog2_model_{model_option}.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = Prog2Model(model_option=model_option, num_layers=num_layers, tokenizer=tokenizer).to(device)
    model.load_state_dict(torch.load(f"prog2_model_{model_option}.pth"))
    model.eval()

    if inference_prompt:
        prompt = inference_prompt
        generated_tokens = model.prompt(prompt, max_seq_len=max_seq_length, temperature=temperature)
        print(generated_tokens)
        generated_text = tokenizer.decode(generated_tokens)
        print(f"Generated Text: '{generated_text}'")
        assert False, "Exiting after inference."

    # Load the test dataset
    test_dataset = TextCompletionDataset(
        file_path=test_path,
        tokenizer=tokenizer,
        max_seq_len=max_seq_length
    )
    
    subset_size = len(test_dataset) // divider
    test_dataset = torch.utils.data.Subset(test_dataset, range(subset_size))
    
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Compute BLEU score
    compute_bleu_score(model, test_dataloader, tokenizer, device)

    # Compute perplexity
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding index
    compute_perplexity(model, test_dataloader, criterion, device)

    prompt = "Which do you prefer? Dogs or cats? "
    generated_tokens = model.prompt(prompt, max_seq_len=max_seq_length, temperature=1)
    print(generated_tokens)
    generated_text = tokenizer.decode(generated_tokens)
    print(f"Generated Text: '{generated_text}'")
