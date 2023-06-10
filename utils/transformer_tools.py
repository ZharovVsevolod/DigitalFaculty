import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

def mask_for_transformer(length):
    # Создаём маску единиц
    full_mask = torch.ones(length, length)
    # Создаём диагональную маску с булевыми значениями
    ignore_mask = torch.tril(full_mask) < 1
    # Заполняем False диагональной маски в "маске единиц" значениями -inf
    full_mask.masked_fill_(ignore_mask, float('-inf'))
    # Остальное - нулями
    full_mask.masked_fill_(~ignore_mask, 0)
    return full_mask

def positional_encoding(max_length, embedding_size):
    # Создаём массив, по которому будут генерироваться синусы и косинусы
    time = np.pi * torch.arange(0, max_length).float()
    freq_dividers = torch.arange(1, embedding_size // 2 + 1).float()
    inputs = time[:, None] / freq_dividers[None, :]
    
    # Берём значения синусов и косинусов в качестве ответа
    result = torch.zeros(max_length, embedding_size)
    result[:, 0::2] = torch.sin(inputs)
    result[:, 1::2] = torch.cos(inputs)
    return result

def lm_cross_entropy(pred, target):
    pred_flat = pred.view(-1, pred.shape[-1])  # BatchSize*TargetLen x VocabSize
    target_flat = target.view(-1)  # BatchSize*TargetLen
    return F.cross_entropy(pred_flat, target_flat, ignore_index=0)

def lm_b_cross_entropy(pred, target):
    pred_flat = pred.view(-1)  # BatchSize*TargetLen x VocabSize
    return F.binary_cross_entropy_with_logits(pred_flat, target)

def lr_scheduler(optimizer):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5, verbose=True)

class Language_Model(nn.Module):
    """
    Основной класс для языковой модели\n
    Может принимать в себя любую форму реализации в качестве внутренней нейросети (backbone)\n
    :param vocab_size: - размер словаря токенов
    :param embedding_size: - размер эмбеддинга
    :param backbone: - нейронная модель для генерации текста
    :param emb_dropout: - размер "прореживания" (dropout) для эмбеддинга
    """
    def __init__(self, vocab_size, embedding_size, backbone, emb_dropout=0.0):
        super().__init__()
        self.embedding_size = embedding_size
        self.embeddings = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.backbone = backbone
        self.out = nn.Linear(embedding_size, vocab_size)
    
    def forward(self, seed_token_ids):
        batch_size, max_in_length = seed_token_ids.shape

        # Создание маски
        seed_padding_mask = seed_token_ids == 0
        dependency_mask = mask_for_transformer(max_in_length).to(seed_token_ids.device)
        
        # Эмбеддинг и позиционное кодирование
        seed_embs = self.embeddings(seed_token_ids)  # Shape = (BatchSize, MaxInLen, EmbSize)
        pos_codes = positional_encoding(max_in_length, self.embedding_size)
        pos_codes = pos_codes.unsqueeze(0).to(seed_embs.device)
        seed_embs = seed_embs + pos_codes
        seed_embs = self.emb_dropout(seed_embs)

        # Shape =  (BatchSize, TargetLen, EmbSize)
        target_features = self.backbone(
            seed_embs,
            mask=dependency_mask,
            src_key_padding_mask=seed_padding_mask
        )
        logits = self.out(target_features)  # Shape =  (BatchSize, TargetLen, VocabSize)
        return logits

class Transformer_Encoder(nn.Module):
    """
    Класс трансформера-энкодера, нужен для транспонирования входных данных
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.impl = nn.TransformerEncoder(*args, **kwargs)
        self.initialize_weights()
    
    def forward(self, src, *args, **kwargs):
        src = src.transpose(0, 1).contiguous()  # Shape = (MaxInLen, BatchSize, EmbSize)
        result = self.impl(src, *args, **kwargs)  # Shape = (TargetLen, BatchSize, EmbSize)
        result = result.transpose(0, 1).contiguous()  # Shape = (BatchSize, TargetLen, EmbSize)
        return result
    
    def initialize_weights(self):
        for param in self.impl.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)


class Language_Discriminator(nn.Module):
    def __init__(self, vocab_size, embedding_size, max_tokens, num_classes=2):
        super().__init__()
        self.emb = embedding_size # 256
        self.max_tokens = max_tokens
        self.embeddings = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.body = nn.Sequential(
            nn.Conv1d(embedding_size, embedding_size//2, 7, padding="same"), # 256, 128, 7
            nn.Conv1d(embedding_size//2, embedding_size//4, 5),
            nn.Conv1d(embedding_size//4, embedding_size//8, 3),
            nn.Conv1d(embedding_size//8, embedding_size//16, 3),
        )
        self.outline = nn.Linear(embedding_size//16, num_classes)
        self.initialize_weights()

    def forward(self, x):
        x = self.embeddings(x) # Shape = (batch, sent_len, emb)

        x = x.transpose(1, 2).contiguous() # Shape = (batch, emb, sent_len)

        x = self.body(x) # Shape = (batch, emb//16, sent_len) 
        x = F.relu(x)
        x, _ = x.max(dim=-1) # Shape = (batch, emb//16)

        x = self.outline(x)
        return x

    def initialize_weights(self):
        for param in self.embeddings.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
                
        for param in self.body.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
        
        for param in self.outline.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

class LSTM_Discriminator(nn.Module):
    """
    Класс, реализующий дискриминатор для генеративно-состязательной сети с архитектурой LSTM
    :param hidden_size: - размер скрытого внутреннего слоя LSTM
    :param embedding_size: - размер эмбеддинга
    :param vocab_size: - размер словаря
    """
    def __init__(self, hidden_size, embedding_size, vocab_size):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.encoder = nn.LSTM(
            input_size=embedding_size, 
            hidden_size=hidden_size, 
            num_layers=1, 
            batch_first=True
        )
        self.predictor = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        output, (hidden, _) = self.encoder(self.embeddings(x))

        hidden = hidden.squeeze(0)
        preds = self.predictor(hidden)
        return preds