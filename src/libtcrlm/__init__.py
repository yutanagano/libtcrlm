from libtcrlm import schema
from typing import Literal


VERSION = "1.0.1"


def setup(species: Literal["homosapiens", "musmusculus"]):
    schema.tcr.setup(species)
