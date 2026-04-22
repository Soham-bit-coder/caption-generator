# model.py — the CNN encoder and LSTM decoder

import torch
import torch.nn as nn
import torchvision.models as models

from config import EMBED_DIM, HIDDEN_DIM, NUM_LAYERS


# ---------------------------------------------------------------------------
# ENCODER — ResNet50 reads the image and outputs a feature vector
# ---------------------------------------------------------------------------

class EncoderCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Load ResNet50 pretrained on ImageNet
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Remove the last classification layer (we don't need 1000-class output)
        # We keep everything except the final FC layer
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])

        # Add our own layer: 2048 → EMBED_DIM (256)
        self.fc = nn.Linear(2048, EMBED_DIM)

        # Freeze ResNet — we don't want to change its weights, just use them
        for param in self.resnet.parameters():
            param.requires_grad = False

    def forward(self, images):
        # images shape: (batch, 3, 224, 224)

        with torch.no_grad():
            features = self.resnet(images)   # (batch, 2048, 1, 1)

        features = features.squeeze(-1).squeeze(-1)  # (batch, 2048)
        features = self.fc(features)                 # (batch, 256)
        return features


# ---------------------------------------------------------------------------
# DECODER — LSTM reads image features and generates words one by one
# ---------------------------------------------------------------------------

class DecoderLSTM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        # Embedding: turns each word index into a vector of size EMBED_DIM
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM)

        # LSTM: takes word embeddings, outputs hidden states
        self.lstm = nn.LSTM(EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, batch_first=True)

        # Project image features (256) to LSTM hidden size (512)
        # This is used to initialize the LSTM's memory with the image
        self.init_hidden = nn.Linear(EMBED_DIM, HIDDEN_DIM)

        # Final layer: hidden state → word probabilities
        self.fc = nn.Linear(HIDDEN_DIM, vocab_size)

        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        """
        features: (batch, 256)       — image summary from encoder
        captions: (seq_len, batch)   — tokenized captions (teacher forcing)
        """
        # captions: (seq_len, batch) → (batch, seq_len)
        captions = captions.transpose(0, 1)

        # Turn word indices into vectors: (batch, seq_len, 256)
        embeddings = self.dropout(self.embedding(captions))

        # Initialize LSTM hidden state using image features
        # h0, c0 shape: (num_layers, batch, hidden_dim)
        h0 = self.init_hidden(features).unsqueeze(0)  # (1, batch, 512)
        c0 = torch.zeros_like(h0)                     # (1, batch, 512)

        # Run LSTM: output shape (batch, seq_len, 512)
        lstm_out, _ = self.lstm(embeddings, (h0, c0))

        # Project to vocab size: (batch, seq_len, vocab_size)
        out = self.fc(lstm_out)
        return out


# ---------------------------------------------------------------------------
# Full model — combines encoder + decoder
# ---------------------------------------------------------------------------

class CaptioningModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = EncoderCNN()
        self.decoder = DecoderLSTM(vocab_size)

    def forward(self, images, captions):
        features = self.encoder(images)       # (batch, 256)
        outputs  = self.decoder(features, captions)  # (batch, seq_len, vocab_size)
        return outputs

    def generate(self, image, vocab, max_len=30):
        """
        Given one image, generate a caption word by word.
        image: (1, 3, 224, 224)
        """
        self.eval()
        with torch.no_grad():
            features = self.encoder(image)   # (1, 256)

            # Start with <SOS> token
            word_idx = torch.tensor([[vocab.word2idx["<SOS>"]]], device=image.device)

            # Initialize LSTM state from image
            h = self.decoder.init_hidden(features).unsqueeze(0)  # (1, 1, 512)
            c = torch.zeros_like(h)

            result = []
            for _ in range(max_len):
                embed    = self.decoder.embedding(word_idx)       # (1, 1, 256)
                out, (h, c) = self.decoder.lstm(embed, (h, c))    # (1, 1, 512)
                logits   = self.decoder.fc(out.squeeze(1))        # (1, vocab_size)
                word_idx = logits.argmax(dim=1, keepdim=True)     # pick best word

                token = word_idx.item()
                if token == vocab.word2idx["<EOS>"]:
                    break
                result.append(token)

        return vocab.decode(result)
