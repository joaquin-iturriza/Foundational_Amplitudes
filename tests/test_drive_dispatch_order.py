"""drive.py dispatch_order: with sweeps of different trial length and fewer slots than wanted trials, the trials
issued come longest first and alternate between sweeps of equal length (main fills site pick's slots in this order)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sweep.drive import dispatch_order


class Sw:                                                            # hashable, as drive.Sweep
    def __init__(self, name, hours):
        self.name, self.hours = name, hours


def test_longest_first_round_robin_under_cap():
    short_a, short_b, long_c, long_d = Sw("a", 0.2), Sw("b", 0.2), Sw("c", 1.0), Sw("d", 1.0)
    want = {short_a: 3, short_b: 3, long_c: 3, long_d: 2}           # config order puts the short sweeps first
    cap = 7                                                          # slots site pick offers, fewer than 11 wanted
    issued = [sw.name for sw in dispatch_order(want)[:cap]]
    assert issued == ["c", "d", "c", "d", "c", "a", "b"], issued
    assert [sw.name for sw in dispatch_order(want)].count("a") == 3 and len(dispatch_order(want)) == 11


if __name__ == "__main__":
    test_longest_first_round_robin_under_cap(); print("ok")
