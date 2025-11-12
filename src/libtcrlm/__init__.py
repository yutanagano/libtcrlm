from libtcrlm import schema
from typing import Literal


def setup(species: Literal["homosapiens", "musmusculus"]):
    schema.tcr.setup(species)
