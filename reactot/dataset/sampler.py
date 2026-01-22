import warnings
from typing import Iterator, List, Optional

import torch

from torch.utils.data import Dataset, RandomSampler
from torch.utils.data.sampler import Sampler, BatchSampler
from torch.utils.data.distributed import DistributedSampler


class DynamicBatchSampler(Sampler):
    r"""Dynamically adds samples to a mini-batch up to a maximum size (either
    based on number of nodes or number of edges). When data samples have a
    wide range in sizes, specifying a mini-batch size in terms of number of
    samples is not ideal and can cause CUDA OOM errors.

    Within the :class:`DynamicBatchSampler`, the number of steps per epoch is
    ambiguous, depending on the order of the samples. By default the
    :meth:`__len__` will be undefined. This is fine for most cases but
    progress bars will be infinite. Alternatively, :obj:`num_steps` can be
    supplied to cap the number of mini-batches produced by the sampler.

    .. code-block:: python

        from torch_geometric.loader import DataLoader, DynamicBatchSampler

        sampler = DynamicBatchSampler(dataset, max_num=10000, mode="node")
        loader = DataLoader(dataset, batch_sampler=sampler, ...)

    Args:
        dataset (Dataset): Dataset to sample from.
        max_num (int): Size of mini-batch to aim for in number of nodes or
            edges.
        mode (str, optional): :obj:`"node"` or :obj:`"edge"` to measure
            batch size. (default: :obj:`"node"`)
        shuffle (bool, optional): If set to :obj:`True`, will have the data
            reshuffled at every epoch. (default: :obj:`False`)
        skip_too_big (bool, optional): If set to :obj:`True`, skip samples
            which cannot fit in a batch by itself. (default: :obj:`False`)
        num_steps (int, optional): The number of mini-batches to draw for a
            single epoch. If set to :obj:`None`, will iterate through all the
            underlying examples, but :meth:`__len__` will be :obj:`None` since
            it is be ambiguous. (default: :obj:`None`)
    """
    def __init__(
        self, 
        dataset: Dataset,
        max_num: int, 
        mode: str = 'node',
        shuffle: bool = False, 
        skip_too_big: bool = False,
        num_steps: Optional[int] = None,
        drop_last: bool = True,
        max_batch: Optional[int] = None,
        ddp: bool = False,
        **kwargs
    ):
        if not isinstance(max_num, int) or max_num <= 0:
            raise ValueError(f"`max_num` should be a positive integer value "
                             "(got {max_num}).")
        self.mode_avail =  ['node', 'node^2']
        self.mode_calc_map = {
            "node": self.node_calc,
            "node^2": self.node_square_calc,
        }
        
        if not mode in self.mode_avail:
            raise ValueError(f"mode {self.mode} is not available.")
        self.mode_calc = self.mode_calc_map[mode]

        if num_steps is None:
            num_steps = len(dataset)

        self.dataset = dataset
        self.num_samples = len(dataset)
        self.max_num = max_num
        self.mode = mode
        self.shuffle = shuffle
        self.skip_too_big = skip_too_big
        self.num_steps = num_steps
        self.drop_last = drop_last
        self.max_batch = max_batch
        if not ddp:
            self.sampler = RandomSampler(
                dataset, 
                generator=torch.Generator().manual_seed(42),
            )
        else:
            self.sampler = DistributedSampler(
                dataset, shuffle=shuffle, seed=42,
            )
        
        self.batch_size = self.max_num // 400
        if self.max_batch is None:
            self.max_batch = len(dataset) // self.batch_size
        
    @staticmethod
    def node_calc(x):
        return x
    
    @staticmethod
    def node_square_calc(x):
        return x ** 2

    def __iter__(self) -> Iterator[List[int]]:
        batch: List[int] = []
        batch_n = 0
        num_batch = 0
        for idx in self.sampler:
            data = self.dataset[idx]
            n = self.mode_calc(data["size_0"].item())

            if len(batch) and batch_n + n > self.max_num:
                # Mini-batch filled
                # print("batch: ", batch)
                yield batch
                batch = []
                batch_n = 0
                num_batch += 1
            
                if (self.max_batch is not None) \
                    and (num_batch > self.max_batch):
                    break 
                
            if n > self.max_num:
                if self.skip_too_big:
                    continue
                else:
                    warnings.warn(
                        f"Size of data sample at index {idx} is larger than"
                        f"{self.max_num} at {self.mode}s (got {n})."
                        "This warning suugests that some systems you have does"
                        "not fit in one GPU."
                    )
            batch.append(idx)
            batch_n += n
        
        if not self.drop_last and len(batch):
            yield batch

    def __len__(self) -> int:
        return self.max_batch