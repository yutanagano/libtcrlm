from libtcrlm.schema import Tcr
from libtcrlm.schema import Pmhc


class TcrPmhcPair:
    def __init__(self, tcr: Tcr, pmhc: Pmhc) -> None:
        self.tcr = tcr
        self.pmhc = pmhc

    def __repr__(self) -> str:
        return f"({self.tcr}) - ({self.pmhc})"
