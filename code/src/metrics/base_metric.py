"""
This class is largely based on the code Felix Heinickel wrote for his Master's Thesis.
(As are most of the other metrics classes)
"""

from abc import ABC, abstractmethod
from typing import Dict

import torch


class Metric(ABC):
    def __init__(self, eos_token_pointer_id: int):
        self.eos_token_pointer_id = eos_token_pointer_id

    @abstractmethod
    def evaluate(self, pred: torch.Tensor, tgt: torch.Tensor):
        raise NotImplementedError

    @abstractmethod
    def get_metrics(self, reset=False):
        raise NotImplementedError
