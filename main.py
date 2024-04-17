import torch
from torch.utils.data import DataLoader

from dataset import *
from transformer import CustomTransformer
from train import train_epoch, val_epoch
from translate import translate_file

import warnings

warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)

src_lang, tgt_lang = 'de', 'en'
tokenizers = dict()
tokenizers["en"] = get_tokenizer(data_dir + "train.de-en.en", "en")
tokenizers["de"] = get_tokenizer(data_dir + "train.de-en.de", "de")

src_vocab_size = tokenizers[src_lang].vocab_size()
tgt_vocab_size = tokenizers[tgt_lang].vocab_size()
d_model = 512
n_heads = 8
dim_feedforward = 512
num_encoder_layers = 3
num_decoder_layers = 3
transformer = CustomTransformer(num_encoder_layers, num_decoder_layers, d_model,
                                n_heads, src_vocab_size, tgt_vocab_size,
                                dim_feedforward).to(device)

bs = 64
train_dataset = TextDataset(data_dir + "train.de-en.en", data_dir + "train.de-en.de", )
train_dataloader = DataLoader(train_dataset, batch_size=bs, collate_fn=collate_fn)
val_dataset = TextDataset(data_dir + "val.de-en.en", data_dir + "val.de-en.de", train=False)
val_dataloader = DataLoader(val_dataset, batch_size=bs, collate_fn=collate_fn)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
scheduler = None
n_epochs = 18
for epoch in range(1, n_epochs + 1):
    train_loss = train_epoch(transformer, optimizer, scheduler, train_dataloader, loss_fn,
                             f'Training {epoch}/{n_epochs}')
    val_loss = val_epoch(transformer, val_dataloader, loss_fn,
                         f'Validating {epoch}/{n_epochs}')
    print(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}")

translate_file(transformer, data_dir + "test1.de-en.de", tokenizers)
