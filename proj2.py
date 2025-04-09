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

def create_tokenizer(input_dir :str):
    """Creates a tokenizer using texts from input_dir and saves it to tokenizer.pkl"""
    # Define the input directory and output model prefix
    model_prefix = "bpe_tokenizer"
    vocab_size = 10000  # You can adjust this based on your needs

    #preprocess input files by adding <eos> tokens at the end of each sentence for each input file
    for filename in os.listdir(input_dir):
        with open(os.path.join(input_dir, filename), "r", encoding="utf-8") as file:
            content = file.read()
        # Add <eos> token at the end of each sentence
        content = content.replace(". ", ". <eos> ")
        with open(os.path.join(input_dir, filename), "w", encoding="utf-8") as file:
            file.write(content)

    # Collect all text files in the input directory
    input_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.txt')]

    # Join file paths into a single string separated by commas
    input_files_str = ",".join(input_files)

    
    # Train the BPE tokenizer
    spm.SentencePieceTrainer.Train(
        input=input_files_str,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe"  # Use Byte Pair Encoding (BPE)
    )

    # Load the trained tokenizer model
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(f"{model_prefix}.model")

    
    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    return tokenizer


class Prog2Model(nn.Module):
    def __init__(self, model_option: str, num_layers: int, tokenizer: spm.SentencePieceProcessor):
        super(Prog2Model, self).__init__()
        self.model_option = model_option
        self.num_layers = num_layers
        self.vocab_size = 10000
        self.tokenizer = tokenizer

        # Define the embedding layer
        self.embedding_layer = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=128)

        # Define the model based on the selected option
        if self.model_option == "RNN":
            self.model = nn.RNN(input_size=128, hidden_size=128, num_layers=self.num_layers, batch_first=True)
        elif self.model_option == "LSTM":
            self.model = nn.LSTM(input_size=128, hidden_size=128, num_layers=self.num_layers, batch_first=True)
        elif self.model_option == "Transformer":
            self.model = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=128, nhead=8), num_layers=self.num_layers
            )
        else:
            raise ValueError("Invalid model option. Choose from 'RNN', 'LSTM', or 'Transformer'.")

        # Define the fully connected layer for output
        self.fc = nn.Linear(128, self.vocab_size)

    def forward(self, tokens, temperature=1.0, train_mode=True):
        # Embed the tokens
        embedded_tokens = self.embedding_layer(tokens)

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
    
    def prompt(self, prompt_text: str, max_seq_len: int = 50):
        """Tokenizes the input text and returns the token IDs."""
        tokens = self.tokenizer.encode(prompt_text, out_type=int)
        while len(tokens) < max_seq_len: #also add eos
            next_token = self(tokens)
            tokens.append(next_token.item())
            if "<eos>" in self.tokenizer.IdToPiece(next_token.item()):
                break
        return tokens
        


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

    for epoch in range(epochs):
        total_loss = 0
        model.train()

        # Training loop
        for batch in train_dataloader:
            input_ids, target_ids = batch
            # print(batch)
            input_ids, target_ids = input_ids.to(device), target_ids.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(input_ids)

                # Reshape outputs and targets for CrossEntropyLoss
            outputs = outputs.view(-1, model.vocab_size)  # Shape: (batch_size * sequence_length, vocab_size)
            target_ids = target_ids.view(-1)  # Shape: (batch_size * sequence_length)


            # Compute loss
            loss = criterion(outputs, target_ids)
            total_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids, target_ids = batch
                input_ids, target_ids = input_ids.to(device), target_ids.to(device)

                outputs = model(input_ids)
                
                # Reshape outputs and targets for CrossEntropyLoss
                outputs = outputs.view(-1, model.vocab_size)  # Shape: (batch_size * sequence_length, vocab_size)
                target_ids = target_ids.view(-1)  # Shape: (batch_size * sequence_length)


                # Compute loss
                loss = criterion(outputs, target_ids)
                val_loss += loss.item()

        val_loss /= len(val_dataloader)
        scheduler.step(val_loss)

        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {total_loss / len(train_dataloader):.4f}, Validation Loss: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_prog2_model.pth")
            print("Model improved and saved.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

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
    # Load the tokenizer
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    max_seq_length = 250
    # Load the dataset
    train_dataset = TextCompletionDataset(
        file_path=r"C:\Users\muslo\Documents\Homework\Foundational AI\CSC7809_FoundationModels\Project2\data\train.jsonl",
        tokenizer=tokenizer,
        max_seq_len=max_seq_length
    )
    
    # Split the dataset into training and validation sets (80-20 split)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

    # Create DataLoaders for training and validation sets
    train_dataloader = DataLoader(train_subset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_subset, batch_size=32, shuffle=False)

    model_option = "Transformer"  # Choose from "RNN", "LSTM", or "Transformer"
    # Initialize the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Prog2Model(model_option=model_option, num_layers=2, tokenizer=tokenizer).to(device)

    # Define the optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Train the model
    train_model(model, train_dataloader, val_dataloader, optimizer, criterion, device, epochs=5)

    # Save the trained model
    torch.save(model.state_dict(), f"prog2_model_{model_option}.pth")