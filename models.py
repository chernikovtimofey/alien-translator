import math
from typing import Iterable
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from tokenizers import Tokenizer
from transformers import BeamSearchScorer
from data import *

class PositionalEncoder(nn.Module):
    """Module that encodes information about position into embedding."""

    def __init__(self, d_model: int, dropout_p: int, max_len: int):
        """
        Makes positional encoding vectors.

        Args:
            d_model (int): Embdedding dimensionality.
            dropout_p (int): Dropout probability.
            max_len (int): Maximum length of sequence of embeddings to encode.
        """

        super().__init__()

        self.dropout = nn.Dropout(dropout_p)

        positions = torch.arange(0, max_len)
        division_term = torch.pow(10000, torch.arange(0, d_model, 2) / d_model)
        fun_arg = positions.view(-1, 1) / division_term.view(1, -1)

        pos_encoding = torch.empty((max_len, d_model))
        pos_encoding[:, 0::2] = torch.sin(fun_arg)
        pos_encoding[:, 1::2] = torch.cos(fun_arg)
        pos_encoding = pos_encoding.unsqueeze(0)
        self.register_buffer('pos_encoding', pos_encoding)

    def forward(self, embedding: torch.Tensor):
        return self.dropout(embedding + self.pos_encoding[:, :embedding.size(1)])

class TranslationTransformer(nn.Module):
    """Wrapper of PyTorch's Transformer model for translation."""

    def __init__(self, num_src_tokens: int, num_tgt_tokens: int, 
                 d_model, nhead, 
                 num_encoder_layers, num_decoder_layers, 
                 dim_feedforward, dropout):
        """
        Initializes all necessary modules.

        Args:
            num_src_tokens (int): Number of tokens of src sequence.
            num_tgt_tokens (int): Number of tokens of tgt sequence.
            d_model (int): The number of expected features in the encoder/decoder inputs.
            nhead (int): The number of heads in the multiheadattention models.
            num_encoder_layers (int): The number of sub-encoder-layers in the encoder.
            num_decoder_layers (int): The number of sub-decoder-layers in the decoder.
            dim_feedforward (int): The dimension of the feedforward network model. 
            dropout (int): The dropout value. 
        """

        MAX_LEN = 500

        super().__init__()

        self.d_model = d_model

        self.src_embedding = nn.Embedding(num_src_tokens, d_model)
        self.tgt_embedding = nn.Embedding(num_tgt_tokens, d_model)

        self.positional_encoder = PositionalEncoder(d_model, dropout, MAX_LEN)

        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True
        )

        self.output_layer = nn.Linear(d_model, num_tgt_tokens)

    def forward(
            self, src: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor = None, pad_token_id: int = 0
    ):
        """
        Args:
            src (torch.Tensor): The src sequence.
            tgt (torch.Tensor): The tgt sequence.
            tgt_mask (torch.Tensor, optional): The additive mask for the tgt. Defaults to a square causal mask for the sequence.
            pad_token_id (int): [PAD] token id.

        Returns:
            torch.Tensor: Prediction scores for each tgt sequence element.
        """

        device = next(self.parameters()).device

        if not tgt_mask:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1), device=device)
        src_padding_mask = TranslationTransformer.generate_padding_mask(src, pad_token_id=pad_token_id, device=device)
        tgt_padding_mask = TranslationTransformer.generate_padding_mask(tgt, pad_token_id=pad_token_id, device=device)

        src = self.src_embedding(src) * math.sqrt(self.d_model)
        tgt = self.tgt_embedding(tgt) * math.sqrt(self.d_model)

        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)

        output = self.transformer(
            src, tgt, 
            tgt_mask=tgt_mask, 
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )
        output = self.output_layer(output)
        return output
    
    def generate_padding_mask(sequences: torch.Tensor, pad_token_id: int = 0, device: Union[torch.device, str] = torch.device('cpu')):
        """
        Generates padding mask of given sequences.

        Args:
            sequences (torch.Tensor): Sequences to generate padding mask on.
            pad_token_id (int, optional): [PAD] token id. Defaults to 0.
            device (Union[torch.device, str]): Device to put the returned mask on. Defaults to CPU.

        Returns:
            torch.Tensor
        """

        return (sequences == pad_token_id).to(device)

def train_loop(
        model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer, 
        loss_fn: nn.Module, verbose: bool = False
    ):
    """
    Performs one epoch of model training.

    Args:
        model (nn.Module): The model to train.
        dataloader (DataLoader): Dataloader to train on.
        optimizer (torch.optim.Optimizer): Optimizer to train a model.
        loss_fn (nn.Module): A loss function to compute the training loss.
        verbose (bool): Whether to show the training process.

    Returns:
        float: An average loss of training all the batches.
    """

    dataset_size = len(dataloader.dataset)

    model.train()

    total_loss = 0
    dataset_processed_size = 0
    for batch_num, (src, tgt) in enumerate(dataloader, start=1):
        input_tgt = tgt[:, :-1]
        target_tgt = tgt[:, 1:]

        scores = model(src, input_tgt)

        loss = loss_fn(scores.transpose(1, 2), target_tgt)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        dataset_processed_size += src.size(0)

        if batch_num % 100 == 0 and verbose:
            print(f'loss: {loss.item():>7f} [{dataset_processed_size:>5}/{dataset_size:>5}])')

    return total_loss / len(dataloader)

def validation_loop(model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module):
    """
    Computes average loss on given dataloader.

    Args:
        model (nn.Module): The model to get scores.
        dataloader (DataLoader): Dataloader on which we want to compute the average loss.
        loss_fn (nn.Module): A loss function.

    Returns:
        float: An average loss.
    """

    model.eval()

    total_loss = 0
    with torch.no_grad():
        for (src, tgt) in dataloader:
            input_tgt = tgt[:, :-1]
            target_tgt = tgt[:, 1:]

            scores = model(src, input_tgt)

            loss = loss_fn(scores.transpose(1, 2), target_tgt)
            total_loss += loss.detach().item()

    return total_loss / len(dataloader)

def fit(
        model: nn.Module, train_dataloader: DataLoader, val_dataloader: DataLoader, 
        optimizer: torch.optim.Optimizer, loss_fn: torch.nn.Module, num_epochs: int, 
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        verbose: bool = False, save_file_path: str = 'saved/model-weights.pth'
):
    """
    Performs model training on given dataloader.
    
    Args:
        model (nn.Module): A model to train.
        train_dataloader (DataLoader): A dataloader to fit.
        val_dataloader (DataLoader): A dataloader to validate.
        optimizer (torch.optim.Optimizer): An optimizer of model's parameters.
        loss_fn (torch.nn.Module): A loss function to compute the training loss.
        num_epochs (int): Number of epochs to perform.
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler. Defaults to None.
        verbose (bool, optional): Whether to show the training process. Defaults to False.
        save_file_path (str, optional): A file path to save model parameters after every epoch. Defaults to 'saved/model_weights.pth'.

    Returns:
        Tuple[List[float], List[float]]: Training loss history and validation loss history.
    """

    train_loss_hist = []
    val_loss_hist = []

    for epoch in range(1, num_epochs + 1):
        print('-' * 25, f'Epoch {epoch}', '-' * 25)

        train_loss = train_loop(model, train_dataloader, optimizer, loss_fn, verbose=verbose)
        torch.save(model.state_dict(), os.path.join(save_file_path))
        train_loss_hist.append(train_loss)
        lr_scheduler.step(train_loss)

        val_loss = validation_loop(model, val_dataloader, loss_fn)
        val_loss_hist.append(val_loss)

        if verbose:
            print(f'Training loss: {train_loss:>7f}')
            print(f'Validation loss: {val_loss:>7f}')
            print(f'Next learning rate: {lr_scheduler.get_last_lr()[0]:>8f}')

    return train_loss_hist, val_loss_hist

def greed_translate(
        model: nn.Module, srcs: Iterable[torch.Tensor], max_length: int = 50,
        pad_token_id: int = 0, bos_token_id: int = 2, eos_token_id: int = 3
):
    """
    Performs greedy search to tranlate the sequence.

    Args:
    model (nn.Module): A model used for translation.
    srcs (Iterable[torch.Tensor]): Batches to translate.
    max_length (int, optional): Maximum length of tranlation sequence. Defaults to 50.
    pad_token_id (int, optional): [PAD] token id. Defaults to 0.
    bos_token_id (int, optional): [BOS] token id. Defaults to 2.
    eos_token_id (int, optional): [EOS] token id. Defaults to 3.
    
    Returns:
        list[tuple]: A list of translation sequences and scores for each batch. Each tuple contains:
            - torch.Tensor: translation of sequences.
            - torch.Tensor: scores of translations.
    """

    model.eval()

    device = next(model.parameters()).device

    with torch.no_grad():
        for src in srcs:
            batch_size = src.size(0)

            tgt = torch.full((batch_size, 1), bos_token_id, device=device)
            seq_score = torch.zeros((batch_size), device=device)

            for _ in range(max_length):
                next_token_scores = model(src, tgt)[:, -1, :]
                next_token_scores[:, pad_token_id] = float('-inf')
                next_token_scores = F.log_softmax(next_token_scores, dim=-1)
                
                next_token_score, next_token_id = torch.max(next_token_scores, dim=-1)

                is_sequence_ended = (tgt[:, -1] == eos_token_id) | (tgt[:, -1] == pad_token_id)
                next_token_id[is_sequence_ended] = pad_token_id
                next_token_score[is_sequence_ended] = 0
                if torch.all(is_sequence_ended):
                    break

                tgt = torch.hstack((tgt, next_token_id.view(-1, 1)))
                seq_score += next_token_score

            seq_score /= torch.count_nonzero(tgt != pad_token_id, dim=-1)

            yield tgt, seq_score

def beam_translate(
        model: nn.Module, srcs: Iterable[torch.Tensor], num_beams: int = 10, max_length: int = 30, 
        pad_token_id: int = 0, bos_token_id: int = 2, eos_token_id: int = 3
):
    """
    Performs beam search to tranlate the sequence.

    Args:
        model (nn.Module): The model used for translation.
        srcs (Iterable[torch.Tensor]): The src batches.
        num_beams (int, optional): Number of beam hypothesis to keep track on. Defaults to 10.
        max_length (int, optional): Maximum length of tranlation sequence. Defaults to 30.
        pad_token_id (int, optional): [PAD] token id. Defaults to 0.
        bos_token_id (int, optional): [BOS] token id. Defaults to 2.
        eos_token_id (int, optional): [EOS] token id. Defaults to 3.

    Returns:
        list[tuple]: A list of translation sequences and scores for each batch. Each tuple contains:
            - torch.Tensor: translation of sequences.
            - torch.Tensor: scores of translations.
    """

    model.eval()

    device = next(model.parameters()).device

    with torch.no_grad():
        for src in srcs:
            batch_size = src.size(0)

            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=num_beams,
                device=device,
                max_length=max_length
            )

            tgt = torch.full((batch_size, 1), bos_token_id, device=device)

            beam_scores = torch.zeros((batch_size), device=device)

            next_token_ids = None
            next_beam_ids = None

            for step in range(1, max_length + 1):  
                next_token_scores = model(src, tgt)[:, -1, :]
                next_token_scores[:, pad_token_id] = float('-inf') 
                next_token_scores = F.log_softmax(next_token_scores, dim=-1)
                
                vocabulary_size = next_token_scores.size(-1)

                next_seq_scores = beam_scores.view(-1, 1) + next_token_scores
                next_seq_scores = next_seq_scores.view(batch_size, -1)

                next_seq_scores, next_token_ids = torch.topk(
                    next_seq_scores, 
                    2*num_beams, 
                    dim=-1, sorted=True
                )

                next_token_beams = (next_token_ids // vocabulary_size)
                next_token_ids = next_token_ids % vocabulary_size

                if step == 1:
                    src = src.unsqueeze(1).repeat(1, num_beams, 1)
                    src = src.view((batch_size * num_beams, -1))

                    tgt = torch.full((batch_size * num_beams, 1), bos_token_id, dtype=torch.long, device=device)

                nexts = beam_scorer.process(
                    tgt, next_seq_scores, next_token_ids, next_token_beams, 
                    pad_token_id=pad_token_id, eos_token_id=eos_token_id
                )
                beam_scores = nexts['next_beam_scores']
                next_token_ids = nexts['next_beam_tokens']
                next_token_beams = nexts['next_beam_indices']

                if beam_scorer.is_done:
                    break

                if step != max_length:
                    tgt = torch.hstack((tgt[next_token_beams], next_token_ids.view(-1, 1)))

            final_seq = beam_scorer.finalize(
                tgt, beam_scores, next_token_ids, next_beam_ids, max_length,
                pad_token_id=pad_token_id, eos_token_id=eos_token_id
            )
            
            yield final_seq['sequences'], final_seq['sequence_scores']

def check_positional_encoder(device: torch.device):
    print('Check PositionalEcoder:')

    DROPOUT_P = 0
    BATCH_SIZE = 2
    SEQ_LEN = 5
    D_MODEL = 10

    positional_encoder = PositionalEncoder(D_MODEL, DROPOUT_P, 100).to(device)
    encoded = positional_encoder(torch.zeros(BATCH_SIZE, SEQ_LEN, D_MODEL, device=device))

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

def check_translation_transformer(device: torch.device):
    print('Check TranslationTransformer:')

    SUBSET = 'train'
    VOCABULARY_SIZE = 5000

    script_dir_path = os.path.dirname(__file__)

    dataset_dir_path = os.path.join(script_dir_path, 'dataset')
    save_dir_path = os.path.join(script_dir_path, 'saved')
    src_tokenizer_save_file_path = os.path.join(save_dir_path, 'src-tokenizer.json')
    dst_tokenizer_save_file_path = os.path.join(save_dir_path, 'dst-tokenizer.json')

    src_tokenizer = Tokenizer.from_file(src_tokenizer_save_file_path)
    dst_tokenizer = Tokenizer.from_file(dst_tokenizer_save_file_path)

    src_transform = Lambda(lambda src : torch.tensor(src_tokenizer.encode(src).ids, device=device))
    dst_transform = Lambda(lambda dst : torch.tensor(dst_tokenizer.encode(dst).ids, device=device))

    dataset = AlienDataset(
        dataset_dir_path, subset=SUBSET, 
        src_transform=src_transform, dst_transform=dst_transform
    )

    batch_sampler = BucketSampler(dataset, is_test=(SUBSET == 'test'), batch_size=4, bucket_size=5, shuffle=False)
    collate_fn = (lambda sequences : bucket_collate_fn(sequences, is_test=(SUBSET == 'test')))
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_fn)

    batch = next(iter(dataloader))

    src, dst = batch

    transformer = TranslationTransformer(
        VOCABULARY_SIZE, VOCABULARY_SIZE, 
        d_model=16, nhead=2, num_encoder_layers=2, num_decoder_layers=2, 
        dim_feedforward=64, dropout=0
    ).to(device)

    print('input:')
    print(src)
    print(dst)

    print('output:')
    out = transformer(src, dst)
    print(out)
    print(out.shape)

def check_transformer_train_loop(device: torch.device):
    print("Check transformer's train_loop:")

    VOCABULARY_SIZE = 5000

    D_MODEL = 32
    N_HEAD = 1
    NUM_ENCODER_LAYERS = 1
    NUM_DECODER_LAYERS = 1
    DIM_FEEDFORWARD = 128
    DROPOUT = 0

    NUM_SAMPLES = 30

    LEARNING_RATE = 0.01
    NUM_EPOCHS = 50

    script_dir_path = os.path.dirname(__file__)

    save_dir_path = os.path.join(script_dir_path, 'saved')

    src_tokenizer_save_file_path = os.path.join(save_dir_path, 'src-tokenizer.json')
    dst_tokenizer_save_file_path = os.path.join(save_dir_path, 'dst-tokenizer.json')
    simple_model_weights_save_file_path = os.path.join(save_dir_path, 'simple-transformer-model-weights.pth')

    dataset_dir_path = os.path.join(script_dir_path, 'dataset')

    src_tokenizer = Tokenizer.from_file(src_tokenizer_save_file_path)
    dst_tokenizer = Tokenizer.from_file(dst_tokenizer_save_file_path)

    src_transform = Lambda(lambda src : torch.tensor(src_tokenizer.encode(src).ids, dtype=torch.long))
    dst_transform = Lambda(lambda dst : torch.tensor(dst_tokenizer.encode(dst).ids, dtype=torch.long))

    dataset = AlienDataset(
        dataset_dir_path, subset='train',
        src_transform=src_transform, dst_transform=dst_transform
    )
    dataset = Subset(dataset, range(NUM_SAMPLES))

    collate_fn = (lambda sequences : bucket_collate_fn(sequences, is_test=False))
    dataloader = DataLoader(dataset, batch_size=NUM_SAMPLES, shuffle=True, collate_fn=collate_fn)

    model = TranslationTransformer(
        VOCABULARY_SIZE, VOCABULARY_SIZE,
        d_model=D_MODEL, nhead=N_HEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS, num_decoder_layers=NUM_DECODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT
    ).to(device)
    
    if os.path.isfile(simple_model_weights_save_file_path):
        model.load_state_dict(torch.load(simple_model_weights_save_file_path, weights_only=True))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(1, NUM_EPOCHS + 1):
            train_loss = train_loop(model, dataloader, optimizer, loss_fn, verbose=False)

            print(f'Epoch {epoch:>3} training loss: {train_loss:>.4f}')

        torch.save(model.state_dict(), simple_model_weights_save_file_path)

def check_greed_translate(device: torch.device):
    print('Check greed_translate:')

    VOCABULARY_SIZE = 5000

    D_MODEL = 32
    N_HEAD = 1
    NUM_ENCODER_LAYERS = 1
    NUM_DECODER_LAYERS = 1
    DIM_FEEDFORWARD = 128
    DROPOUT = 0

    NUM_SAMPLES = 10

    MAX_LENGTH = 30

    script_dir_path = os.path.dirname(__file__)

    save_dir_path = os.path.join(script_dir_path, 'saved')
    src_tokenizer_save_file_path = os.path.join(save_dir_path, 'src-tokenizer.json')
    dst_tokenizer_save_file_path = os.path.join(save_dir_path, 'dst-tokenizer.json')
    model_weights_save_file_path = os.path.join(save_dir_path, 'simple-transformer-model-weights.pth')

    dataset_dir_path = os.path.join(script_dir_path, 'dataset')

    src_tokenizer = Tokenizer.from_file(src_tokenizer_save_file_path)
    dst_tokenizer = Tokenizer.from_file(dst_tokenizer_save_file_path)

    src_transform = Lambda(lambda src : torch.tensor(src_tokenizer.encode(src).ids, dtype=torch.long).to(device))
    dst_transform = Lambda(lambda dst : torch.tensor(dst_tokenizer.encode(dst).ids, dtype=torch.long).to(device))

    dataset = AlienDataset(
        dataset_dir_path, subset='train',
        src_transform=src_transform, dst_transform=dst_transform
    )
    dataset = Subset(dataset, range(NUM_SAMPLES))

    collate_fn = (lambda sequences : bucket_collate_fn(sequences, is_test=False))
    dataloader = DataLoader(dataset, batch_size=NUM_SAMPLES, shuffle=True, collate_fn=collate_fn)

    model = TranslationTransformer(
        VOCABULARY_SIZE, VOCABULARY_SIZE,
        d_model=D_MODEL, nhead=N_HEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS, num_decoder_layers=NUM_DECODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT
    ).to(device)
    
    model.load_state_dict(torch.load(model_weights_save_file_path, weights_only=True, map_location=device))

    with torch.no_grad():
        collate_fn = (lambda sequences : bucket_collate_fn(sequences, is_test=False)[0])
        dataloader = DataLoader(dataset, batch_size=NUM_SAMPLES, shuffle=False, collate_fn=collate_fn)

        seqs, scores = next(greed_translate(model, dataloader, max_length=MAX_LENGTH))
        for (dst, seq, score) in zip(dataset, seqs, scores):
            dst = dst[1]
            print('actural translation:', dst_tokenizer.decode(dst.tolist()))
            print('greed translation:', dst_tokenizer.decode(seq.tolist()))
            print('score:', score.item())
            print('-' * 10)

def check_beam_translate(device: torch.device):
    print('Check beam_translate:')

    VOCABULARY_SIZE = 5000

    D_MODEL = 32
    N_HEAD = 1
    NUM_ENCODER_LAYERS = 1
    NUM_DECODER_LAYERS = 1
    DIM_FEEDFORWARD = 128
    DROPOUT = 0

    NUM_SAMPLES = 10

    NUM_BEAMS = 10

    MAX_LENGTH = 30

    script_dir_path = os.path.dirname(__file__)

    save_dir_path = os.path.join(script_dir_path, 'saved')
    src_tokenizer_save_file_path = os.path.join(save_dir_path, 'src-tokenizer.json')
    dst_tokenizer_save_file_path = os.path.join(save_dir_path, 'dst-tokenizer.json')
    model_weights_save_file_path = os.path.join(save_dir_path, 'simple-transformer-model-weights.pth')

    dataset_dir_path = os.path.join(script_dir_path, 'dataset')

    src_tokenizer = Tokenizer.from_file(src_tokenizer_save_file_path)
    dst_tokenizer = Tokenizer.from_file(dst_tokenizer_save_file_path)

    src_transform = Lambda(lambda src : torch.tensor(src_tokenizer.encode(src).ids, dtype=torch.long).to(device))
    dst_transform = Lambda(lambda dst : torch.tensor(dst_tokenizer.encode(dst).ids, dtype=torch.long).to(device))

    dataset = AlienDataset(dataset_dir_path, subset='train',
                           src_transform=src_transform, dst_transform=dst_transform)
    dataset = Subset(dataset, range(NUM_SAMPLES))

    collate_fn = (lambda sequences : bucket_collate_fn(sequences, is_test=False))
    dataloader = DataLoader(dataset, batch_size=NUM_SAMPLES, shuffle=True, collate_fn=collate_fn)

    model = TranslationTransformer(
        VOCABULARY_SIZE, VOCABULARY_SIZE,
        d_model=D_MODEL, nhead=N_HEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS, num_decoder_layers=NUM_DECODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT
    ).to(device)
    
    model.load_state_dict(torch.load(model_weights_save_file_path, weights_only=True, map_location=device))

    with torch.no_grad():
        collate_fn = (lambda sequences : bucket_collate_fn(sequences, is_test=False)[0])
        dataloader = DataLoader(dataset, batch_size=NUM_SAMPLES, shuffle=False, collate_fn=collate_fn)

        seqs, scores = next(beam_translate(model, dataloader, num_beams=NUM_BEAMS, max_length=MAX_LENGTH))
        for (dst, seq, score) in zip(dataset, seqs, scores):
            dst = dst[1]
            print('actural translation:', dst_tokenizer.decode(dst.tolist()))
            print('beam translation:', dst_tokenizer.decode(seq.tolist()))
            print('score:', score.item())
            print('-' * 10)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    check_positional_encoder(device)
    check_translation_transformer(device)
    check_transformer_train_loop(device)
    check_greed_translate(device)
    check_beam_translate(device)