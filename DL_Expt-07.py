# ================================================================
# Deep Learning Lab Experiment 07
# CNN + RNN based Image Captioning
# ================================================================
import os
import random
import string
import numpy as np
import pandas as pd
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm

# ================================================================
# Device configuration
# ================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# ================================================================
# Step 1 – Paths (update these)
# ================================================================
images_dir = r"C:\Users\dilip\datasets\Deep Learning 5th sem\lab expts\image captions\Images"           # ← path to your image folder
captions_file = r"C:\Users\dilip\datasets\Deep Learning 5th sem\lab expts\image captions\captions.txt"    # ← path to captions.txt

# ================================================================
# Step 2 – Load and preprocess captions
# ================================================================
df = pd.read_csv(captions_file, header=0, names=["image", "caption"], encoding='utf-8')

def clean_caption(c):
    c = c.lower().strip()
    c = c.translate(str.maketrans('', '', string.punctuation))
    return " ".join(c.split())

df["caption"] = df["caption"].astype(str).apply(clean_caption)

print(f"Total caption rows: {len(df)}")
print(f"Unique images: {df['image'].nunique()}")

# ================================================================
# Step 3 – Build vocabulary
# ================================================================
min_word_freq = 5
max_vocab_size = 10000

tokens = []
for cap in df["caption"]:
    tokens.extend(cap.split())

word_counts = Counter(tokens)
vocab_words = [w for w, c in word_counts.items() if c >= min_word_freq]
vocab_words = sorted(vocab_words, key=lambda w: -word_counts[w])[:max_vocab_size]

PAD, SOS, EOS, UNK = "<pad>", "<sos>", "<eos>", "<unk>"
itos = [PAD, SOS, EOS, UNK] + vocab_words
stoi = {w: i for i, w in enumerate(itos)}
vocab_size = len(itos)
print("Vocab size:", vocab_size)

max_caption_len = max(len(c.split()) for c in df["caption"]) + 2
print("Max caption length:", max_caption_len)

def encode_caption(caption, stoi, max_len=None):
    idxs = [stoi.get(w, stoi[UNK]) for w in caption.split()]
    seq = [stoi[SOS]] + idxs + [stoi[EOS]]
    if max_len:
        if len(seq) < max_len:
            seq += [stoi[PAD]] * (max_len - len(seq))
        else:
            seq = seq[:max_len]
    return seq

# ================================================================
# Step 4 – Dataset class
# ================================================================
class ImageCaptionDataset(Dataset):
    def __init__(self, df, images_dir, stoi, transform=None, max_len=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.stoi = stoi
        self.transform = transform
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.images_dir, row["image"])
        caption = row["caption"]

        if not os.path.exists(img_path):
            img = Image.new("RGB", (224, 224), color=(0, 0, 0))
        else:
            img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        seq = encode_caption(caption, self.stoi, self.max_len)
        seq = torch.tensor(seq, dtype=torch.long)
        return img, seq

# ================================================================
# Step 5 – Data transforms and loaders
# ================================================================
IMG_SIZE = 224
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Split train / val
indices = list(range(len(df)))
random.seed(42)
random.shuffle(indices)
split = int(0.9 * len(indices))
train_df = df.iloc[indices[:split]]
val_df = df.iloc[indices[split:]]

train_dataset = ImageCaptionDataset(train_df, images_dir, stoi, transform, max_caption_len)
val_dataset = ImageCaptionDataset(val_df, images_dir, stoi, transform, max_caption_len)

BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

print(f"Train size: {len(train_dataset)} | Val size: {len(val_dataset)}")

# ================================================================
# Step 6 – Model definitions
# ================================================================
class EncoderCNN(nn.Module):
    def __init__(self, embed_size=512, train_cnn=False):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        modules = list(resnet.children())[:-2]
        self.backbone = nn.Sequential(*modules)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(embed_size)

        if not train_cnn:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, images):
        feats = self.backbone(images)
        pooled = self.avgpool(feats).squeeze()
        embeddings = self.relu(self.embed(pooled))
        embeddings = self.bn(embeddings)
        return embeddings

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, dropout=0.0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embed(captions[:, :-1])
        features = features.unsqueeze(1)
        inputs = torch.cat((features, embeddings), dim=1)
        out, _ = self.lstm(inputs)
        out = self.fc(out)
        return out

    def sample(self, features, max_len=30, sos_idx=1):
        sampled_ids, states = [], None
        inputs = features.unsqueeze(1)
        for _ in range(max_len):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.fc(hiddens.squeeze(1))
            _, predicted = outputs.max(1)
            sampled_ids.append(predicted.item())
            if predicted.item() == stoi["<eos>"]:
                break
            inputs = self.embed(predicted).unsqueeze(1)
        return sampled_ids

# ================================================================
# Step 7 – Initialize, loss, optimizer
# ================================================================
embed_size, hidden_size, num_layers = 512, 512, 1
encoder = EncoderCNN(embed_size=embed_size, train_cnn=False).to(device)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=stoi["<pad>"])
params = list(decoder.parameters()) + list(encoder.embed.parameters()) + list(encoder.bn.parameters())
optimizer = optim.Adam(params, lr=1e-4)

# ================================================================
# Step 8 – Training & validation
# ================================================================
def train_one_epoch(epoch):
    encoder.train()
    decoder.train()
    total_loss = 0.0
    for imgs, caps in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        imgs, caps = imgs.to(device), caps.to(device)
        optimizer.zero_grad()
        feats = encoder(imgs)
        outputs = decoder(feats, caps)
        B, seq_len, vocab = outputs.size()
        loss = criterion(outputs.view(B * seq_len, vocab), caps.view(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(decoder.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def validate():
    encoder.eval()
    decoder.eval()
    total_loss = 0.0
    with torch.no_grad():
        for imgs, caps in val_loader:
            imgs, caps = imgs.to(device), caps.to(device)
            feats = encoder(imgs)
            outputs = decoder(feats, caps)
            B, seq_len, vocab = outputs.size()
            loss = criterion(outputs.view(B * seq_len, vocab), caps.view(-1))
            total_loss += loss.item()
    return total_loss / len(val_loader)

# ================================================================
# Step 9 – Main loop (Windows-safe)
# ================================================================
def main():
    EPOCHS = 6
    best_val_loss = float("inf")
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(epoch)
        val_loss = validate()
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "stoi": stoi,
                "itos": itos
            }, "checkpoints/best_caption_model.pth")
            print("✅ Saved best model.")

    print("\nTraining complete!")

# ================================================================
# Step 10 – Entry point
# ================================================================
if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
