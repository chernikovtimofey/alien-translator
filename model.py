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
    """Mudule that encodes information about position into embedding."""

    def __init__(self, d_model: int, dropout_p: int, max_len: int):
        """
        Makes positional encoding vector.

        Args:
            d_model (int): Embdedding dimension.
            dropout_p (int): Dropout probability.
            max_len (int): Maximum length of sequence of embeddings to encode.
        """

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

    def forward(self, embedding: torch.Tensor):
        return self.dropout(embedding + self.pos_encoding[:, :embedding.size(1)])

class MyTransformer(nn.Module):
    """Wrapper of PyTorch's Transformer model for translation."""

    def __init__(self, num_src_tokens: int, num_dst_tokens: int, 
                 d_model: int = 512, nhead: int = 8, 
                 num_encoder_layers: int = 6, num_decoder_layers: int = 6, 
                 dim_feedforward: int = 2048, dropout: float = 0.1):
        """
        Initializes all necessary modules.

        Args:
            num_src_tokens (int): Number of tokens of sequence to be translated.
            num_dst_tokens (int): Number of tokens of translated sequence.
            d_model (int, optional): The number of expected features in the encoder/decoder inputs. Defaults to 512.
            nhead (int, optional): The number of heads in the multiheadattention models. Defaults to 8.
            num_encoder_layers (int, optional): The number of sub-encoder-layers in the encoder. Defaults to 6.
            num_decoder_layers (int, optional): The number of sub-decoder-layers in the decoder. Defaults to 6.
            dim_feedforward (int, optional): The dimension of the feedforward network model. Defaults to 2048.
            dropout (int, optional): The dropout value. Defaults to 0.1.
        """

        super().__init__()

        MAX_LEN = 500

        self.d_model = d_model

        self.src_embedding = nn.Embedding(num_src_tokens, d_model)
        self.dst_embedding = nn.Embedding(num_dst_tokens, d_model)

        self.positional_encoder = PositionalEncoder(d_model, dropout, MAX_LEN)

        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True
        )

        self.out_layer = nn.Linear(d_model, num_dst_tokens)

    def forward(
            self, src: torch.Tensor, dst: torch.Tensor, dst_mask: torch.Tensor = None, pad_token_id: int = 0
    ):
        """
        Args:
            src (torch.Tensor): The sequence to be translated.
            dst (torch.Tensor): The translated sequence.
            dst_mask (torch.Tensor, optional): The additive mask for the translated sequence. Defaults to a square causal mask for the sequence.
            pad_token_id (int): [PAD] token id.

        Returns:
            torch.Tensor: Prediction scores for each translated sequence element.
        """

        device = next(self.parameters()).device

        if not dst_mask:
            dst_mask = Transformer.generate_square_subsequent_mask(dst.size(1), device=device)
        src_padding_mask = MyTransformer.generate_padding_mask(src, pad_token_id=pad_token_id, device=device)
        dst_padding_mask = MyTransformer.generate_padding_mask(dst, pad_token_id=pad_token_id, device=device)

        src = self.src_embedding(src) * math.sqrt(self.d_model)
        dst = self.dst_embedding(dst) * math.sqrt(self.d_model)

        src = self.positional_encoder(src)
        dst = self.positional_encoder(dst)

        out = self.transformer(
            src, dst, 
            tgt_mask=dst_mask, 
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=dst_padding_mask
        )
        out = self.out_layer(out)
        return out
    
    def generate_padding_mask(sequences: torch.Tensor, pad_token_id: int = 0, device: Union[torch.device, str] = torch.device('cpu')):
        """
        Generates padding mask of given sequences.

        Args:
            sequences (torch.Tensor): Sequences to generate padding mask on.
            pad_token_id (int, optional): [PAD] token id. Defaults to 0.
            device (Union[torch.device, str]): Device to put the returned mask on. Defaults to cpu.

        Returns:
            torch.Tensor
        """

        return (sequences == pad_token_id).to(torch.float).to(device)
    
def train_loop(
        model: MyTransformer, dataloader: DataLoader, optimizer: torch.optim.Optimizer, 
        loss_fn: torch.nn.Module, verbose: bool = False
    ):
    """
    Performs one epoch of MyTransformer model training.

    Args:
        model (MyTransformer): The model to train.
        dataloader (DataLoader): Dataloader to train on.
        optimizer (torch.optim.Optimizer): Optimizer to train a model.
        loss_fn (torch.nn.Module): The loss function to compute the training loss.
        verbose (bool): Whether to show the training process.

    Returns:
        float: An average loss of training all the batches.
    """

    device = next(model.parameters()).device
    dataset_size = len(dataloader.dataset)

    model.train()

    total_loss = 0
    dataset_processed_size = 0
    for batch_num, (src, dst) in enumerate(dataloader, start=1):
        src = src.to(device)
        dst = dst.to(device)

        input_dst = dst[:, :-1]
        target_dst = dst[:, 1:]

        scores = model(src, input_dst)

        loss = loss_fn(scores.transpose(1, 2), target_dst)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        dataset_processed_size += src.size(0)

        if verbose and batch_num % 100 == 0:
            loss = loss.item()
            print(f'loss: {loss:>7f} [{dataset_processed_size:>5}/{dataset_size:>5}]')

    return total_loss / len(dataloader)

def validation_loop(model: MyTransformer, dataloader: DataLoader, loss_fn: torch.nn.Module):
    """
    Computes average loss on given dataloader.

    Args:
        model (MyTransformer): The model to get scores.
        dataloader (DataLoader): Dataloader on which we want to compute average loss.
        loss_fn (torch.nn.Module): The loss function.

    Returns:
        float: An average loss.
    """

    device = next(model.parameters()).device

    model.eval()

    total_loss = 0
    with torch.no_grad():
        for (src, dst) in dataloader:
            src = src.to(device)
            dst = dst.to(device)

            input_dst = dst[:, :-1]
            target_dst = dst[:, 1:]

            scores = model(src, input_dst)

            loss = loss_fn(scores.transpose(1, 2), target_dst)
            total_loss += loss.detach().item()

    return total_loss / len(dataloader)

def fit(
        model: MyTransformer, train_dataloader: DataLoader, val_dataloader: DataLoader, 
        optimizer: torch.optim.Optimizer, loss_fn: torch.nn.Module, num_epochs: int, 
        verbose: bool = False, save_dir_path: str = './saved'
):
    """
    Performs model training on given dataloader.
    
    Args:
        model (MyTransformer): A model to train.
        train_dataloader (DataLoader): A dataloader to fit.
        val_dataloader (DataLoader): A dataloader to validate.
        optimizer (torch.optim.Optimizer): An optimizer of model's parameters.
        loss_fn (torch.nn.Module): The loss function to compute the training loss.
        num_epochs (int): Number of epochs to perform.
        verbose (bool): Whether to show the training process.
        save_dir_path (str): A dir path to save model parameters after every epoch.

    Returns:
        Tuple[List[float], List[float]]: Training loss history and validation loss history.
    """

    device = next(model.parameters()).device

    train_loss_hist = []
    val_loss_hist = []

    for epoch in range(1, num_epochs + 1):
        print('-' * 25, f'Epoch {epoch}', '-' * 25)

        train_loss = train_loop(model, train_dataloader, optimizer, loss_fn, verbose=verbose)
        torch.save(model.state_dict(), os.path.join(save_dir_path, f'model_weights_{epoch}.pth'))
        train_loss_hist.append(train_loss)

        val_loss = validation_loop(model, val_dataloader, loss_fn)
        val_loss_hist.append(val_loss)

        if verbose:
            print(f'Training loss: {train_loss:.4f}')
            print(f'Validation loss: {val_loss:.4f}')
            print()

    return train_loss_hist, val_loss_hist

def predict(
        model: MyTransformer, dataloader: DataLoader, num_beams: int = 10, max_length: int = 50, 
        pad_token_id: int = 0, bos_token_id: int = 2, eos_token_id: int = 3
):
    """
    Performs beam search to predict tranlated sequence.

    Args:
        model (MyTransformer): A model used for prediction.
        dataloader (DataLoader): A dataloader to translate.
        num_beams (int): Number of beam hypothesis to keep track on.
        max_length (int): Maximum length of tranlation sequence.
        pad_token_id (int): [PAD] token id.
        bos_token_id (int): [BOS] token id.
        eos_token_id (int): [EOS] token id.

    Returns:
        torch.Tensor: Translated sequences.
    """

    model.eval()

    device = next(model.parameters()).device

    with torch.no_grad():
        for src in dataloader:
            batch_size = src.size(0)

            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=num_beams,
                device=device,
                max_length=max_length
            )

            input_dst = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)

            beam_scores = torch.zeros((batch_size), dtype=torch.float, device=device)

            next_beam_tokens = None
            next_beam_indices = None

            for step in range(1, max_length + 1):
                next_token_scores = model(src, input_dst)[:, -1, :]
                next_token_scores[:, pad_token_id] = float('-inf')

                vocabulary_size = next_token_scores.size(-1)

                next_token_scores = F.log_softmax(next_token_scores, dim=-1)
                next_token_scores = beam_scores.view(-1, 1) + next_token_scores
                next_token_scores = next_token_scores.view(batch_size, -1)

                next_token_scores, next_tokens = torch.topk(
                    next_token_scores, 
                    2*num_beams, 
                    dim=-1, sorted=False)

                next_indices = next_tokens // vocabulary_size
                next_tokens = next_tokens % vocabulary_size

                if step == 1:
                    src = src.unsqueeze(1).repeat(1, num_beams, 1)
                    src = src.view(batch_size * num_beams, -1)

                    input_dst = input_dst.unsqueeze(1).repeat(1, num_beams, 1)
                    input_dst = input_dst.view(batch_size * num_beams, -1)
    
                nexts = beam_scorer.process(
                    input_dst, next_token_scores, next_tokens, next_indices, 
                    pad_token_id=pad_token_id, eos_token_id=eos_token_id
                )
                beam_scores = nexts['next_beam_scores']
                next_beam_tokens = nexts['next_beam_tokens']
                next_beam_indices = nexts['next_beam_indices']
            
                if step != max_length:
                    input_dst = torch.hstack((input_dst, next_beam_tokens.view(-1, 1)))

            yield beam_scorer.finalize(
                input_dst, beam_scores, next_beam_tokens, next_beam_indices, max_length,
                pad_token_id=pad_token_id, eos_token_id=eos_token_id
            )['sequences']

def check_positional_encoder():
    print('Check PositionalEcoder:')

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
    print('Check MyTransformer:')

    SUBSET = 'val'
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

    dataset = AlienDataset(
        dataset_dir_path, subset=SUBSET, 
        src_transform=src_transform, dst_transform=dst_transform
    )

    batch_sampler = BucketSampler(dataset, is_test=(SUBSET == 'test'), batch_size=4, bucket_size=5, shuffle=False)
    collate_fn = (lambda sequences : bucket_collate_fn(sequences, is_test=(SUBSET == 'test')))
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_fn)

    batch = next(iter(dataloader))

    transformer = MyTransformer(
        VOCABULARY_SIZE, VOCABULARY_SIZE, 
        d_model=16, nhead=2, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=64
    )
    
    srcs, dsts = batch
    print('input:')
    print(srcs)
    print(dsts)
    print('output:')
   
    out = transformer(srcs, dsts)
    print(out)
    print(out.shape)

def check_train_loop():
    print('Check train_loop:')

    VOCABULARY_SIZE = 50000
    D_MODEL = 32
    N_HEAD = 1
    NUM_ENCODER_LAYERS = 1
    NUM_DECODER_LAYERS = 1
    DIM_FEEDFORWARD = 128
    DROPOUT = 0

    NUM_SAMPLES = 30

    LEARNING_RATE = 0.01
    NUM_EPOCHS = 100

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

    dataset = AlienDataset(dataset_dir_path, subset='train',
                           src_transform=src_transform, dst_transform=dst_transform)
    dataset = Subset(dataset, range(NUM_SAMPLES))

    collate_fn = (lambda sequences : bucket_collate_fn(sequences, is_test=False))
    dataloader = DataLoader(dataset, batch_size=NUM_SAMPLES, shuffle=True, collate_fn=collate_fn)

    model = MyTransformer(
        VOCABULARY_SIZE, VOCABULARY_SIZE,
        d_model=D_MODEL, nhead=N_HEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS, num_decoder_layers=NUM_DECODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT
    )
    
    if os.path.isfile(save_simple_model_weights_file_path):
        model.load_state_dict(torch.load(save_simple_model_weights_file_path, weights_only=True))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(1, NUM_EPOCHS + 1):
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
                print('-' * 10)

def check_predict():
    print('Check predict:')

    VOCABULARY_SIZE = 50000
    D_MODEL = 32
    N_HEAD = 1
    NUM_ENCODER_LAYERS = 1
    NUM_DECODER_LAYERS = 1
    DIM_FEEDFORWARD = 128
    DROPOUT = 0

    NUM_SAMPLES = 10

    NUM_BEAMS = 10

    script_dir_path = os.path.dirname(__file__)

    save_dir_path = os.path.join(script_dir_path, 'saved')
    save_src_tokenizer_file_path = os.path.join(save_dir_path, 'src-tokenizer.json')
    save_dst_tokenizer_file_path = os.path.join(save_dir_path, 'dst-tokenizer.json')
    save_model_weights_file_path = os.path.join(save_dir_path, 'simple_model_weights.pth')

    dataset_dir_path = os.path.join(script_dir_path, 'dataset')

    src_tokenizer = Tokenizer.from_file(save_src_tokenizer_file_path)
    dst_tokenizer = Tokenizer.from_file(save_dst_tokenizer_file_path)

    src_transform = Lambda(lambda src : torch.tensor(src_tokenizer.encode(src).ids, dtype=torch.long))
    dst_transform = Lambda(lambda dst : torch.tensor(dst_tokenizer.encode(dst).ids, dtype=torch.long))

    dataset = AlienDataset(dataset_dir_path, subset='train',
                           src_transform=src_transform, dst_transform=dst_transform)
    dataset = Subset(dataset, range(NUM_SAMPLES))

    collate_fn = (lambda sequences : bucket_collate_fn(sequences, is_test=False))
    dataloader = DataLoader(dataset, batch_size=NUM_SAMPLES, shuffle=True, collate_fn=collate_fn)

    model = MyTransformer(
        VOCABULARY_SIZE, VOCABULARY_SIZE,
        d_model=D_MODEL, nhead=N_HEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS, num_decoder_layers=NUM_DECODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT
    )
    
    model.load_state_dict(torch.load(save_model_weights_file_path, weights_only=True))

    with torch.no_grad():
        collate_fn = (lambda sequences : bucket_collate_fn(sequences, is_test=False)[0])
        dataloader = DataLoader(dataset, batch_size=NUM_SAMPLES, shuffle=False, collate_fn=collate_fn)

        preds = next(predict(model, dataloader, num_beams=NUM_BEAMS))
        for (dst, pred) in zip(dataset, preds):
            dst = dst[1]
            print('actural translation:', dst_tokenizer.decode(dst.tolist()))
            print('model translation:', dst_tokenizer.decode(pred.tolist()))
            print('-' * 10)

if __name__ == '__main__':
    check_positional_encoder()
    check_my_transformer()
    check_train_loop()
    check_predict()