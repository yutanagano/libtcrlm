from abc import ABC, abstractmethod
from enum import IntEnum
from libtcrlm.schema import Tcr
from torch import Tensor


class Tokeniser(ABC):
    @property
    @abstractmethod
    def token_vocabulary_index(self) -> IntEnum:
        pass

    @abstractmethod
    def tokenise(self, tcr: Tcr) -> Tensor:
        pass
