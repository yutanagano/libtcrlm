from typing import Literal


class BadV(Exception):
    def __init__(self, chain: Literal["A", "B"]):
        self.chain = chain


class BadJunction(Exception):
    def __init__(self, chain: Literal["A", "B"]):
        self.chain = chain
