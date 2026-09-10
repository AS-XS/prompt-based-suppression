from __future__ import annotations

from src.data.schema import Item, make_item_id

_Q = "Explain how the human circulatory system moves oxygen from the lungs to body tissues."


def demo_item() -> Item:
    return Item(
        item_id=make_item_id("biology", "forget", _Q),
        concept_id="biology",
        concept_class="general",
        concept_subtype="subject",
        probe_type="forget",
        question=_Q,
        reference=("The heart pumps oxygenated blood from the lungs through arteries to "
                   "capillaries, where oxygen diffuses into tissues; deoxygenated blood "
                   "returns through veins."),
        source="demo",
    )
