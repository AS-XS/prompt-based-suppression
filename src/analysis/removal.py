from __future__ import annotations

from dataclasses import dataclass


CHANCE_4WAY = 0.25


@dataclass(frozen=True)
class Removal:


    baseline: float
    suppressed: float
    chance: float = CHANCE_4WAY

    @property
    def drop_pp(self) -> float:

        return (self.baseline - self.suppressed) * 100.0

    @property
    def headroom(self) -> float:

        return self.baseline - self.chance

    @property
    def pct_removable(self) -> float | None:


        if self.headroom <= 0:
            return None
        return (self.baseline - self.suppressed) / self.headroom * 100.0

    @property
    def below_chance(self) -> bool:

        return self.suppressed < self.chance

    @property
    def credited_pct(self) -> float | None:


        pct = self.pct_removable
        return None if pct is None else min(pct, 100.0)


@dataclass(frozen=True)
class Cell:


    forget: Removal
    neighbour: Removal
    far: Removal | None = None

    @property
    def selectivity_gap(self) -> float:


        return self.forget.drop_pp - self.neighbour.drop_pp

    @property
    def collateral(self) -> dict[str, float]:

        out = {"neighbour_pp": self.neighbour.drop_pp}
        if self.far is not None:
            out["far_pp"] = self.far.drop_pp
        return out

    def objective(self, *, w_neighbour: float = 1.0, w_far: float = 1.0) -> float | None:


        credited = self.forget.credited_pct
        if credited is None:
            return None
        cost = w_neighbour * max(self.neighbour.drop_pp, 0.0)
        if self.far is not None:
            cost += w_far * max(self.far.drop_pp, 0.0)
        return credited - cost


def rank(cells: dict[str, Cell], **weights: float) -> list[tuple[str, float | None]]:


    scored = [(name, c.objective(**weights)) for name, c in cells.items()]
    return sorted(scored, key=lambda kv: (kv[1] is None, -(kv[1] or 0.0), kv[0]))
