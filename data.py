import os
import json
import random
from typing import Callable
from typing import Union
from tokenizers import Tokenizer
from tokenizers import normalizers
from tokenizers import pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import TemplateProcessing
import torch
from torch.utils.data import Sampler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchvision.transforms import Lambda

def make_src_tokenizer(srcs: list[str], vocab_size: int, show_progress: bool = False):
    """
    Makes BPE tokenizer for strings to be translated.

    Args:
        srcs (list[str]): A list of strings to be translated.
        vocab_size (int): The size of the final vocabulary, including all tokens and alphabet.
        show_progress (bool, optional): Whether to show progress bars while training.

    Returns:
        tokenizers.Tokenizer
    """

    tokenizer = Tokenizer(BPE(unk_token='[UNK]'))
    trainer = BpeTrainer(
        vocab_size=vocab_size, show_progress=show_progress, special_tokens=['[PAD]', '[UNK]', '[BOS]', '[EOS]']
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = TemplateProcessing(
        single='[BOS] $A [EOS]',
        special_tokens=[('[BOS]', 2), ('[EOS]', 3)]
    )

    tokenizer.train_from_iterator(srcs, trainer)

    return tokenizer

def make_dst_tokenizer(dsts: list[str], vocab_size: int, show_progress: bool = False):
    """
    Makes BPE tokenizer for translation strings.

    Args:
        dsts (list[str]): A list of translation strings.
        vocab_size (int): The size of the final vocabulary, including all tokens and alphabet.
        show_progress (bool, optional): Whether to show progress bars while training.

    Returns:
        tokenizers.Tokenizer
    """

    tokenizer = Tokenizer(BPE(unk_token='[UNK]'))
    trainer = BpeTrainer(
        vocab_size=vocab_size, show_progress=show_progress, special_tokens=['[PAD]', '[UNK]', '[BOS]', '[EOS]']
    )
    tokenizer.normalizer = normalizers.Lowercase()
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.Whitespace(), pre_tokenizers.Digits(individual_digits=True)]
    )
    tokenizer.post_processor = TemplateProcessing(
        single='[BOS] $A [EOS]', 
        special_tokens=[('[BOS]', 2), ('[EOS]', 3)]
    )   

    tokenizer.train_from_iterator(dsts, trainer)

    return tokenizer

class AlienDataset(Dataset):
    """Dataset class that loads Alien dataset."""

    def __init__(
            self, dataset_dir_path: str, subset: str = 'test', 
            src_transform: Callable[[str], any] = None, dst_transform: Callable[[str], any] = None
    ):
        """
        Loads the dataset.

        Args:
            dataset_dir_path (str): Path to Alien dataset directory.
            subset (str, optional): Can take values "train", "val" and "test" depending on the subset you want to load. Defaults to "test". 
            src_transform (Callable[[str], any], optional): Transformation of string to be translated. Defaults to None. 
            dst_transform (Callable[[str], any], optional): Transformation of translation string. Defaults to None.

        Raises:
            ValueError: Raised if subset value is not valid.
        """

        self.subset = subset
        self.src_transform = src_transform
        self.dst_transform = dst_transform

        dataset_file_path = None
        if subset == 'test':
            dataset_file_path = os.path.join(dataset_dir_path, 'test_no_reference')
        elif subset == 'train':
            dataset_file_path = os.path.join(dataset_dir_path, 'train')
        elif subset == 'val':
            dataset_file_path = os.path.join(dataset_dir_path, 'val')
        else:
            raise ValueError('Wrong subset name')

        self.srcs = []
        self.dsts = []
        with open(dataset_file_path, 'r') as dataset_file:
            for line in dataset_file:
                sample = json.loads(line)
                self.srcs.append(sample['src'])
                if subset != 'test':
                    self.dsts.append(sample['dst'])

    def __len__(self):
        return len(self.srcs)
    
    def __getitem__(self, idx: int):
        src = self.srcs[idx]
        if self.src_transform:
            src = self.src_transform(src)
    
        if self.subset == 'test':
            return src
        else:
            dst = self.dsts[idx]
            if self.dst_transform:
                dst = self.dst_transform(dst)

            return src, dst
        
class BucketSampler(Sampler):
    """
    Batch sampler that puts in the same batch only sequences in the same bucket.
    
    All the sequences are divided into buckets. Two sequences are in the same bucket 
    if their lengths are within the range of the bucket. Two sequences from different buckets 
    can't be in the same batch though if two sequences are in the same bucket doesn't necessary mean 
    they end up in the same batch.
    """

    def __init__(
        self, dataset: Dataset, is_test: bool, 
        batch_size: int = 64, bucket_size: int = 5, shuffle: bool = False
    ):
        """
        Generates buckets.

        Args:
            dataset (Dataset): Dataset from which we want to sample.
            is_test (bool): Whether the dataset subset is test or not.
            batch_size (int, optional): Batch size. Defaults to 64.
            bucket_size (int, optional): Size of range of a single bucket.
            shuffle (bool, optional): Whether to shuffle the dataset. Defaults to False.
        """

        self.dataset = dataset
        self.is_test = is_test
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.buckets = self._make_buckets(bucket_size)

        self.length = 0
        for bucket in self.buckets:
            self.length += len(bucket) // self.batch_size + (len(bucket) % self.batch_size != 0)

    def __iter__(self):
        batches = []
        for bucket in self.buckets:
            if self.shuffle:
                random.shuffle(bucket)

            for idx in range(0, len(bucket), self.batch_size):
                batches.append(bucket[idx : min(idx + self.batch_size, len(bucket))])

        if self.shuffle:
            random.shuffle(batches)

        return iter(batches)
    
    def __len__(self):
        return self.length

    def _make_buckets(self, bucket_size: int):
        buckets = {}
        for idx, seq in enumerate(self.dataset):
            if not self.is_test:
                seq = seq[0]
            bucket_idx = len(seq) // bucket_size
            if bucket_idx not in buckets:
                buckets[bucket_idx] = []
            buckets[bucket_idx].append(idx)

        return list(buckets.values())
    
def bucket_collate_fn(
    sequences: list[Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]], is_test: bool, 
    padding_value: int = 0
):
    """
    Turns a list of sequences into a single batch. 

    Args:
        sequences (list[Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]]): List of variable length sequences or a list of tuples of two variable length sequences.
        is_test (bool): Whether "sequences" is a list of sequences or a list of tuples. 
        padding_value (int, optional): Value for padded elements. Defaults to 0.
        padding_side (str, optional): The side to pad the sequences on. Defaults to "right".

    Returns:
        torch.Tensor: Batch of sequences.
    """

    if is_test:
        return pad_sequence(sequences, batch_first=True, padding_value=padding_value)
    else:
        pad_sequence_ = \
            (lambda sequences : pad_sequence(sequences, batch_first=True, padding_value=padding_value))
        return tuple(map(pad_sequence_, zip(*sequences)))
    
def check_make_srcs_tokenizer():
    print('Check make srcs_tokenizer:')

    VOCABULARY_SIZE = 50000

    script_dir_path = os.path.dirname(__file__)
    dataset_file_path = os.path.join(script_dir_path, 'dataset/train')

    save_dir_path = os.path.join(script_dir_path, 'saved')
    save_tokenizer_file_path = os.path.join(save_dir_path, 'src-tokenizer.json')

    srcs = []
    with open(dataset_file_path, 'r') as data_file:
        for line in data_file:
            line_data = json.loads(line)
            srcs.append(line_data['src'])   

    tokenizer = None
    if os.path.isfile(save_tokenizer_file_path):
        tokenizer = Tokenizer.from_file(save_tokenizer_file_path)
    else:
        tokenizer = make_src_tokenizer(srcs, VOCABULARY_SIZE, show_progress=True)
        os.makedirs(save_dir_path, exist_ok=True)
        tokenizer.save(save_tokenizer_file_path)
    
    for src in srcs[:10]:
        print(src)
        encoded = tokenizer.encode(src)
        print(encoded.tokens)
        print(encoded.ids)
        print('-' * 10)
        
def check_make_dists_tokenizer():
    print('Check make_dists_tokenizer:')

    VOCABULARY_SIZE = 50000

    script_dir_path = os.path.dirname(__file__)
    dataset_file_path = os.path.join(script_dir_path, 'dataset/train')

    save_dir_path = os.path.join(script_dir_path, 'saved')
    save_tokenizer_file_path = os.path.join(save_dir_path, 'dst-tokenizer.json')

    dsts = []
    with open(dataset_file_path, 'r') as data_file:
        for line in data_file:
            line_data = json.loads(line)
            dsts.append(line_data['dst'])   

    if os.path.isfile(save_tokenizer_file_path):
        tokenizer = Tokenizer.from_file(save_tokenizer_file_path)
    else:
        tokenizer = make_dst_tokenizer(dsts, VOCABULARY_SIZE, show_progress=True)
        os.makedirs(save_dir_path, exist_ok=True)
        tokenizer.save(save_tokenizer_file_path)

    for dst in dsts[:10]:
        print(dst)
        encoded = tokenizer.encode(dst)
        print(encoded.tokens)
        print(encoded.ids)
        print('-' * 10)

def check_alien_dataset():
    print('Check AlienDataset:')

    NUM_TEXTS = 10
    SUBSET = 'train'

    script_dir_path = os.path.dirname(__file__)
    saved_dir_path = os.path.join(script_dir_path, 'saved')
    saved_src_tokenizer_file_path = os.path.join(saved_dir_path, 'src-tokenizer.json')
    saved_dst_tokenizer_file_path = os.path.join(saved_dir_path, 'dst-tokenizer.json')

    dataset_dir_path = os.path.join(script_dir_path, 'dataset')

    src_tokenizer = Tokenizer.from_file(saved_src_tokenizer_file_path)
    dst_tokenizer = Tokenizer.from_file(saved_dst_tokenizer_file_path)

    src_transform = Lambda(lambda src : src_tokenizer.encode(src).tokens)
    dst_transform = Lambda(lambda dst : dst_tokenizer.encode(dst).tokens)

    dataset = AlienDataset(dataset_dir_path, subset=SUBSET, 
                           src_transform=src_transform, dst_transform=dst_transform)
    for idx in range(NUM_TEXTS):
        if SUBSET == 'test':
            src = dataset[idx]
            print(src)
        else:
            src, dst = dataset[idx]
            print(src)
            print(dst)
        print('-' * 10)

def check_bucket_sampler():
    print('Check bucket_sampler:')
    
    SUBSET = 'test'
    BATCH_SIZE = 4
    BUCKET_SIZE = 5
    SHUFFLE = False
    NUM_BATCHES = 1

    script_dir_path = os.path.dirname(__file__)
    dataset_dir_path = os.path.join(script_dir_path, 'dataset')
    saved_dir_path = os.path.join(script_dir_path, 'saved')
    saved_src_tokenizer_file_path = os.path.join(saved_dir_path, 'src-tokenizer.json')
    saved_dst_tokenizer_file_path = os.path.join(saved_dir_path, 'dst-tokenizer.json')

    src_tokenizer = Tokenizer.from_file(saved_src_tokenizer_file_path)
    dst_tokenizer = Tokenizer.from_file(saved_dst_tokenizer_file_path)

    src_transform = Lambda(lambda src : torch.tensor(src_tokenizer.encode(src).ids))
    dst_transform = Lambda(lambda dst : torch.tensor(dst_tokenizer.encode(dst).ids))

    dataset = AlienDataset(dataset_dir_path, subset=SUBSET, 
                           src_transform=src_transform, dst_transform=dst_transform)

    batch_sampler = BucketSampler(dataset, is_test=(SUBSET == 'test'), 
                                  batch_size=BATCH_SIZE, bucket_size=BUCKET_SIZE, shuffle=SHUFFLE)
    collate_fn = (lambda sequences : bucket_collate_fn(sequences, is_test=(SUBSET == 'test')))
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_fn)

    for count, batch in enumerate(dataloader):
        if count == NUM_BATCHES:
            break

        if SUBSET == 'test':
            print(batch)
            print(batch.shape)
        else:
            src_batch, dst_batch = batch
            print(src_batch)
            print(src_batch.shape)
            print('-' * 10)
            print(dst_batch)
            print(dst_batch.shape)
            print('-' * 10)

if __name__ == '__main__':
    check_make_dists_tokenizer()
    check_make_srcs_tokenizer()
    check_alien_dataset()
    check_bucket_sampler()