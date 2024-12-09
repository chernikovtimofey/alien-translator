import math
import torch
from torch import nn
from torch.nn import Transformer
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
from data import *

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, dropout_p, max_len):
        super().__init__()

        self.dropout = nn.Dropout(dropout_p)

        positions = torch.arange(0, max_len, dtype=torch.float)
        division_term = torch.pow(10000, torch.arange(0, d_model, 2) / d_model)
        fun_arg = positions.view(-1, 1) / division_term.view(1, -1)

        pos_encoding = torch.empty((max_len, d_model))
        pos_encoding[:, 0::2] = torch.sin(fun_arg)
        pos_encoding[:, 1::2] = torch.cos(fun_arg)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pos_encoding', pos_encoding)

    def forward(self, embeddings):
        return self.dropout(embeddings + self.pos_encoding[:embeddings.size(0)])

class MyTransformer(nn.Module):
    def __init__(self, num_inp_tokens, num_out_tokens, 
                 d_model=512, nhead=8, 
                 num_encoder_layers=6, num_decoder_layers=6, 
                 dim_feedforward=2048, dropout=0.1):
        super().__init__()

        MAX_LEN = 500

        self.d_model = d_model

        self.src_embedding = nn.Embedding(num_inp_tokens, d_model)
        self.dst_embedding = nn.Embedding(num_out_tokens, d_model)

        self.positional_encoder = PositionalEncoder(d_model, dropout, MAX_LEN)

        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward, dropout=dropout
        )
        self.out_layer = nn.Linear(d_model, num_out_tokens)

    def forward(self, srcs, dsts, dst_mask=None, src_key_padding_mask=None, dst_key_padding_mask=None):
        srcs = self.src_embedding(srcs) * math.sqrt(self.d_model)
        dsts = self.dst_embedding(dsts) * math.sqrt(self.d_model)

        srcs = self.positional_encoder(srcs)
        dsts = self.positional_encoder(dsts)

        out = self.transformer(
            srcs, dsts, 
            tgt_mask=dst_mask, 
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=dst_key_padding_mask
        )
        out = self.out_layer(out)
        return out
    
    def generate_padding_mask(sequences, pad_token=0):
        return (sequences == pad_token).transpose(0, 1)

def check_positional_encoder():
    DROPOUT_P = 0
    SEQ_LEN = 5
    BATCH_SIZE = 2
    D_MODEL = 10

    positional_encoder = PositionalEncoder(D_MODEL, DROPOUT_P, 100)
    encoded = positional_encoder(torch.zeros(SEQ_LEN, BATCH_SIZE, D_MODEL))

    for batch_num in range(BATCH_SIZE):
        for pos in range(SEQ_LEN):
            for idx in range(D_MODEL):
                correct = None
                if idx % 2 == 0:
                    correct = math.sin(pos / 10000**(idx / D_MODEL))
                else:
                    correct = math.cos(pos / 10000**((idx - 1) / D_MODEL))

                print(batch_num, pos, idx)
                print('correct:', correct)
                print('output:', encoded[pos, batch_num, idx].item())
                print('diff:', correct - encoded[pos, batch_num, idx].item())
                print('-' * 10)

def check_my_transformer():
    SUBSAMPLING = 'val'
    VOCABULARY_SIZE = 20000

    script_dir_path = os.path.dirname(__file__)
    dataset_dir_path = os.path.join(script_dir_path, 'dataset')
    save_dir_path = os.path.join(script_dir_path, 'saved')
    saved_src_tokenizer_file_path = os.path.join(save_dir_path, 'src-tokenizer.json')
    saved_dst_tokenizer_file_path = os.path.join(save_dir_path, 'dst-tokenizer.json')

    src_tokenizer = Tokenizer.from_file(saved_src_tokenizer_file_path)
    dst_tokenizer = Tokenizer.from_file(saved_dst_tokenizer_file_path)

    src_transform = Lambda(lambda src : torch.tensor(src_tokenizer.encode(src).ids))
    dst_transform = Lambda(lambda dst : torch.tensor(dst_tokenizer.encode(dst).ids))

    dataset = AlienDataset(dataset_dir_path, subsampling=SUBSAMPLING, 
                           src_transform=src_transform, dst_transform=dst_transform)

    batch_sampler = BucketSampler(dataset, is_test=(SUBSAMPLING == 'test'), batch_size=4, bucket_size=5, shuffle=False)
    collate_fn = (lambda sequences : bucket_collate_fn(sequences, is_test=(SUBSAMPLING == 'test')))
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_fn)

    batch = next(iter(dataloader))

    transformer = MyTransformer(VOCABULARY_SIZE, VOCABULARY_SIZE, 
                                d_model=16, nhead=2, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=64)
    
    srcs, dsts = batch
    print('input:')
    print(srcs)
    print(dsts)
    print('output:')
    dst_msk = Transformer.generate_square_subsequent_mask(dsts.size(0))
    src_key_padding_mask = MyTransformer.generate_padding_mask(srcs)
    dst_key_padding_mask = MyTransformer.generate_padding_mask(dsts)
    print(transformer(srcs, dsts, dst_mask=dst_msk, src_key_padding_mask=src_key_padding_mask, dst_key_padding_mask=dst_key_padding_mask))

if __name__ == '__main__':
    # check_positional_encoder()
    # check_my_transformer()
    pass