from sentencepiece import SentencePieceTrainer, SentencePieceProcessor
import os
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch import tensor


def get_tokenizer(input_file, language):
    if not os.path.isfile(language + 'bpe' + '.model'):
        SentencePieceTrainer.train(
            input=input_file, vocab_size=4000,
            model_type="bpe", model_prefix=language + 'bpe',
            normalization_rule_name="nmt_nfkc_cf",
            pad_id=PAD_IDX, bos_id=BOS_IDX, eos_id=EOS_IDX, unk_id=UNK_IDX)
    return SentencePieceProcessor(model_file=language + 'bpe' + '.model')


class TextDataset(Dataset):
    def __init__(self, en_file, de_file, train=True, max_length=512):
        self.tokenizers = {"en": SentencePieceProcessor(model_file='enbpe.model'),
                           "de": SentencePieceProcessor(model_file='debpe.model')}
        with open(en_file, encoding="utf-8") as file:
            en_texts = file.readlines()

        with open(de_file, encoding="utf-8") as file:
            de_texts = file.readlines()
        self.en_texts = en_texts
        self.de_texts = de_texts
        self.en_indices = self.tokenizers["en"].encode(self.en_texts)
        self.de_indices = self.tokenizers["de"].encode(self.de_texts)

        self.max_length = max_length

    def __len__(self):
        return len(self.en_indices)

    def __getitem__(self, item: int):
        en_encoded = [BOS_IDX] + self.en_indices[item][:self.max_length - 2] + [EOS_IDX]
        de_encoded = [BOS_IDX] + self.de_indices[item][:self.max_length - 2] + [EOS_IDX]
        return de_encoded, en_encoded


def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(tensor(src_sample))
        tgt_batch.append(tensor(tgt_sample))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch


UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
data_dir = "./bhw2-data/data/"
