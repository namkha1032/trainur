import torch
from torch.utils.data import DataLoader
import sys
from dataclasses import dataclass, asdict, fields
import random
import numpy as np
from logab import log_init

@dataclass
class Trainur:
    device: any ="cuda:0"
    dtype: int = 32
    epoch: int = 1
    batch_size: int = 2
    effective_batch_size: int = 32
    accumulated_steps: int = None
    num_workers: int = 4
    prefetch_factor: int = 8

    def __post_init__(self):
        self.accumulated_steps = self.effective_batch_size // self.batch_size
        attr_old = asdict(self)
        
        result_dict = dict()

        for cls in reversed(self.__class__.__mro__):
            if hasattr(cls, '__annotations__'):
                result_dict[cls.__name__] = dict()
                for attr in cls.__annotations__:
                    if attr in attr_old:
                        result_dict[cls.__name__][attr] = attr_old[attr]
                        attr_old.pop(attr, None)

        for cls_name, attr_list in result_dict.items():
            print(f"{cls_name}:")
            for attr_name, attr_value in attr_list.items():
                print(f"\t{attr_name}: {attr_value}")
            print(f"{'-'*50}\n")

        self.dtype = torch.float32 if self.dtype == 32 else torch.bfloat16
        self.logger = log_init()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        dataclass(cls)
        pass

    def fix_seed(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def count_trainable(self, model):
        total = 0
        trainable = 0
        for p in model.parameters():
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()
        result = f"Trainable {trainable} / {total} ({trainable/total*100:.4f})%"
        return result

    def transfer_tensor(self, tensor):
        return {key: value.to(device=self.device, dtype=self.dtype if value.dtype == torch.float32 else value.dtype) if isinstance(value, torch.Tensor) else value for key, value in tensor.items()}
    
    def create_dataloader(self, dataset, drop_last=True, is_shuffle=True):
        if 'debugpy' in sys.modules:
            num_workers = 0
            prefetch_factor = None
            persistent_workers = False
        else:
            num_workers = self.num_workers
            prefetch_factor = self.prefetch_factor
            persistent_workers = True if num_workers > 0 else False
        dataloader = DataLoader(
            dataset, 
            shuffle=is_shuffle, 
            batch_size=self.batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory=True,
            drop_last=drop_last,
        )
        return dataloader
    