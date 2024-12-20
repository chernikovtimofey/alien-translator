import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Transformer
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from tokenizers import Tokenizer
from transformers import BeamSearchScorer
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
        pos_encoding = pos_encoding.unsqueeze(0)
        self.register_buffer('pos_encoding', pos_encoding)

    def forward(self, embeddings):
        return self.dropout(embeddings + self.pos_encoding[:, :embeddings.size(1)])

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
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True
        )
        self.out_layer = nn.Linear(d_model, num_out_tokens)

    def forward(self, srcs, dsts, dst_mask=None, src_key_padding_mask=None, dst_key_padding_mask=None):
        device = next(self.parameters()).device
        if not dst_mask:
            dst_mask = Transformer.generate_square_subsequent_mask(dsts.size(1), device=device)
        if not src_key_padding_mask:
            src_key_padding_mask = MyTransformer.generate_padding_mask(srcs, device=device)
        if not dst_key_padding_mask:
            dst_key_padding_mask = MyTransformer.generate_padding_mask(dsts, device=device)

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
    
    def generate_padding_mask(sequences, pad_token=0, device=torch.device('cpu')):
        return (sequences == pad_token).to(torch.float).to(device)
    
def train_loop(model, dataloader, optimizer, loss_fn, verbose=False, device=torch.device('cpu')):
    dataset_size = len(dataloader.dataset)

    model.train()

    total_loss = 0
    dataset_watched_size = 0
    for batch_num, (srcs, dsts) in enumerate(dataloader, start=1):
        srcs = srcs.to(device)
        dsts = dsts.to(device)

        input_dsts = dsts[:, :-1]
        target_dsts = dsts[:, 1:]

        scores = model(srcs, input_dsts)

        loss = loss_fn(scores.transpose(1, 2), target_dsts)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        dataset_watched_size += srcs.size(0)

        if verbose and batch_num % 100 == 0:
            loss = loss.item()
            print(f'loss: {loss:>7f} [{dataset_watched_size:>5}/{dataset_size:>5}]')

    return total_loss / len(dataloader)

def validation_loop(model, dataloader, loss_fn, device=torch.device('cpu')):
    model.eval()

    total_loss = 0
    with torch.no_grad():
        for (srcs, dsts) in dataloader:
            srcs = srcs.to(device)
            dsts = dsts.to(device)

            dsts_input = dsts[:, :-1]
            dsts_target = dsts[:, 1:]

            dst_mask = Transformer.generate_square_subsequent_mask(dsts_input.size(1), device=device)
            src_key_padding_mask = MyTransformer.generate_padding_mask(srcs, device=device)
            dst_key_padding_mask = MyTransformer.generate_padding_mask(dsts_input, device=device)

            scores = model(srcs, dsts_input, dst_mask=dst_mask,
                          src_key_padding_mask=src_key_padding_mask,
                          dst_key_padding_mask=dst_key_padding_mask)

            loss = loss_fn(scores.transpose(1, 2), dsts_target)
            total_loss += loss.detach().item()

    return total_loss / len(dataloader)

def fit(model, train_dataloader, val_dataloader, optimizer, loss_fn, epochs, verbose=False, device=torch.device('cpu')):
    train_loss_hist = []
    val_loss_hist = []

    for epoch in range(1, epochs + 1):
        print('-' * 25, f'Epoch {epoch}', '-' * 25)

        train_loss = train_loop(model, train_dataloader, optimizer, loss_fn, verbose=verbose, device=device)
        torch.save(model.state_dict(), 'model_weights.pth')
        train_loss_hist.append(train_loss)

        val_loss = validation_loop(model, val_dataloader, loss_fn, device=device)
        val_loss_hist.append(val_loss)

        print(f'Training loss: {train_loss:.4f}')
        print(f'Validation loss: {val_loss:.4f}')
        print()

    return train_loss_hist, val_loss_hist

def predict(model, dataloader, num_beams=10, max_length=50, pad_token_id=0, bos_token_id=2, eos_token_id=3):
    with torch.no_grad():
        model.eval()

        device = next(model.parameters()).device

        for srcs in dataloader:
            batch_size = srcs.size(0)

            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=num_beams,
                device=device,
                max_length=max_length
            )

            input_dsts = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)

            beam_scores = torch.zeros((batch_size), dtype=torch.float, device=device)

            next_beam_tokens = None
            next_beam_indices = None


            for step in range(1, max_length + 1):
                next_token_scores = model(srcs, input_dsts)[:, -1, :]
                next_token_scores[:, pad_token_id] = float('-inf')

                vocabulary_size = next_token_scores.size(-1)

                next_token_scores = F.log_softmax(next_token_scores, dim=-1)
                next_token_scores = beam_scores.view(-1, 1) + next_token_scores
                next_token_scores = next_token_scores.view(batch_size, -1)

                next_token_scores, next_tokens = torch.topk(
                    next_token_scores, 
                    num_beams if step == max_length else 2*num_beams, 
                    dim=-1, sorted=False)

                next_indices = next_tokens // vocabulary_size
                next_tokens = next_tokens % vocabulary_size

                if step == 1:
                        srcs = srcs.unsqueeze(1).repeat(1, num_beams, 1)
                        srcs = srcs.view(batch_size * num_beams, -1)

                        input_dsts = input_dsts.unsqueeze(1).repeat(1, num_beams, 1)
                        input_dsts = input_dsts.view(batch_size * num_beams, -1)
    
                nexts = beam_scorer.process(
                    input_dsts, next_token_scores, next_tokens, next_indices, 
                    pad_token_id=pad_token_id, eos_token_id=eos_token_id
                )
                beam_scores = nexts['next_beam_scores']
                next_beam_tokens = nexts['next_beam_tokens']
                next_beam_indices = nexts['next_beam_indices']
            
                if step != max_length:
                    input_dsts = torch.hstack((input_dsts, next_beam_tokens.view(-1, 1)))

            yield beam_scorer.finalize(
                input_dsts, beam_scores, next_beam_tokens, next_beam_indices, max_length,
                pad_token_id=pad_token_id, eos_token_id=eos_token_id
            )['sequences']

def check_positional_encoder():
    DROPOUT_P = 0
    BATCH_SIZE = 2
    SEQ_LEN = 5
    D_MODEL = 10

    positional_encoder = PositionalEncoder(D_MODEL, DROPOUT_P, 100)
    encoded = positional_encoder(torch.zeros(BATCH_SIZE, SEQ_LEN, D_MODEL))

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
                print('output:', encoded[batch_num, pos, idx].item())
                print('diff:', correct - encoded[batch_num, pos, idx].item())
                print('-' * 10)

def check_my_transformer():
    SUBSAMPLING = 'val'
    VOCABULARY_SIZE = 50000

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
   
    out = transformer(srcs, dsts)
    print(out)
    print(out.shape)

def check_train_loop():
    VOCABULARY_SIZE = 50000
    D_MODEL = 32
    N_HEAD = 1
    NUM_ENCODER_LAYERS = 1
    NUM_DECODER_LAYERS = 1
    DIM_FEEDFORWARD = 128
    DROPOUT = 0

    LEARNING_RATE = 0.01
    EPOCHS = 100

    script_dir_path = os.path.dirname(__file__)

    save_dir_path = os.path.join(script_dir_path, 'saved')
    save_src_tokenizer_file_path = os.path.join(save_dir_path, 'src-tokenizer.json')
    save_dst_tokenizer_file_path = os.path.join(save_dir_path, 'dst-tokenizer.json')
    save_simple_model_weights_file_path = os.path.join(save_dir_path, 'simple_model_weights.pth')

    dataset_dir_path = os.path.join(script_dir_path, 'dataset')

    src_tokenizer = Tokenizer.from_file(save_src_tokenizer_file_path)
    dst_tokenizer = Tokenizer.from_file(save_dst_tokenizer_file_path)

    src_transform = Lambda(lambda src : torch.tensor(src_tokenizer.encode(src).ids, dtype=torch.long))
    dst_transform = Lambda(lambda dst : torch.tensor(dst_tokenizer.encode(dst).ids, dtype=torch.long))

    dataset = AlienDataset(dataset_dir_path, subsampling='train',
                           src_transform=src_transform, dst_transform=dst_transform)
    dataset = Subset(dataset, range(10))

    collate_fn = (lambda sequences : bucket_collate_fn(sequences, is_test=False))
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True, collate_fn=collate_fn)

    model = MyTransformer(VOCABULARY_SIZE, VOCABULARY_SIZE,
                          d_model=D_MODEL, nhead=N_HEAD,
                          num_encoder_layers=NUM_ENCODER_LAYERS, num_decoder_layers=NUM_DECODER_LAYERS,
                          dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT)
    
    if os.path.isfile(save_simple_model_weights_file_path):
        model.load_state_dict(torch.load(save_simple_model_weights_file_path, weights_only=True))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(1, EPOCHS + 1):
            train_loss = train_loop(model, dataloader, optimizer, loss_fn, verbose=False)

            print(f'Epoch {epoch:3} training loss: {train_loss:.4f}')

        torch.save(model.state_dict(), save_simple_model_weights_file_path)

    with torch.no_grad():
        model.eval()

        for (srcs, dsts) in dataloader:
            dsts_input = dsts[:, :-1]

            scores = model(srcs, dsts_input)
            preds = torch.argmax(scores, dim=-1)

            for (dst, pred) in zip(dsts, preds):
                print('actural translation:', dst_tokenizer.decode(dst.tolist()))
                print('model translation:', dst_tokenizer.decode(pred.tolist()))

def check_predict():
    VOCABULARY_SIZE = 50000
    D_MODEL = 32
    N_HEAD = 1
    NUM_ENCODER_LAYERS = 1
    NUM_DECODER_LAYERS = 1
    DIM_FEEDFORWARD = 128
    DROPOUT = 0

    LEARNING_RATE = 0.01
    EPOCHS = 100

    script_dir_path = os.path.dirname(__file__)

    save_dir_path = os.path.join(script_dir_path, 'saved')
    save_src_tokenizer_file_path = os.path.join(save_dir_path, 'src-tokenizer.json')
    save_dst_tokenizer_file_path = os.path.join(save_dir_path, 'dst-tokenizer.json')
    save_simple_model_weights_file_path = os.path.join(save_dir_path, 'simple_model_weights.pth')

    dataset_dir_path = os.path.join(script_dir_path, 'dataset')

    src_tokenizer = Tokenizer.from_file(save_src_tokenizer_file_path)
    dst_tokenizer = Tokenizer.from_file(save_dst_tokenizer_file_path)

    src_transform = Lambda(lambda src : torch.tensor(src_tokenizer.encode(src).ids, dtype=torch.long))
    dst_transform = Lambda(lambda dst : torch.tensor(dst_tokenizer.encode(dst).ids, dtype=torch.long))

    dataset = AlienDataset(dataset_dir_path, subsampling='train',
                           src_transform=src_transform, dst_transform=dst_transform)
    dataset = Subset(dataset, range(10))

    collate_fn = (lambda sequences : bucket_collate_fn(sequences, is_test=False))
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True, collate_fn=collate_fn)

    model = MyTransformer(VOCABULARY_SIZE, VOCABULARY_SIZE,
                          d_model=D_MODEL, nhead=N_HEAD,
                          num_encoder_layers=NUM_ENCODER_LAYERS, num_decoder_layers=NUM_DECODER_LAYERS,
                          dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT)
    
    if os.path.isfile(save_simple_model_weights_file_path):
        model.load_state_dict(torch.load(save_simple_model_weights_file_path, weights_only=True))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(1, EPOCHS + 1):
            train_loss = train_loop(model, dataloader, optimizer, loss_fn, verbose=False)

            print(f'Epoch {epoch:3} training loss: {train_loss:.4f}')

        torch.save(model.state_dict(), save_simple_model_weights_file_path)

    with torch.no_grad():
        collate_fn = (lambda sequences : bucket_collate_fn(sequences, is_test=False)[0])
        dataloader = DataLoader(dataset, batch_size=10, shuffle=False, collate_fn=collate_fn)

        preds = next(predict(model, dataloader, num_beams=2))
        for (dst, pred) in zip(dataset, preds):
            dst = dst[1]
            print('actural translation:', dst_tokenizer.decode(dst.tolist()))
            print('model translation:', dst_tokenizer.decode(pred.tolist()))

if __name__ == '__main__':
    check_positional_encoder()
    check_my_transformer()
    check_train_loop()
    check_predict()