import torch
from train import generate_square_subsequent_mask


def decode(model, src, src_mask, max_len):
    src = src.to(device)
    src_mask = src_mask.to(device)
    memory = model.encode(src, src_mask)
    ys = torch.tensor([BOS_IDX]).reshape(-1, 1).type(torch.long).to(device)
    for i in range(max_len - 1):
        memory = memory.to(device)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).to(device)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.linear(out[:, -1])
        next_word = torch.tensor(torch.argmax(prob, dim=1).item()).reshape(-1, 1).type(torch.long).to(device)
        ys = torch.cat([ys, next_word], dim=0)
        if next_word == EOS_IDX:
            break
    return ys


def translate(model, src_sentence, tokenizers):
    model.eval()
    src = torch.tensor([BOS_IDX] + tokenizers[src_lang].encode(src_sentence) + [EOS_IDX]).reshape(-1, 1)  # view
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = decode(model, src, src_mask, max_len=num_tokens + 5).flatten()
    list_tgt_tokens = list(tgt_tokens.cpu().numpy())
    list_tgt_tokens = [int(x) for x in list_tgt_tokens]
    return tokenizers[tgt_lang].decode(list_tgt_tokens)


def translate_file(transformer, file_name, tokenizers):
    with open(file_name, encoding="utf-8") as file:
        texts = file.readlines()
    with open("ans2.de-en.en", "w", encoding="utf-8") as f:
        for text in texts:
            f.write(translate(transformer, text, tokenizers) + '\n')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
src_lang, tgt_lang = 'de', 'en'
