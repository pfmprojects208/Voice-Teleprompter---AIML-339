"""Tracks script position from ground-truth chunk transition timestamps.

`data/labels/*.json` (written by `dataset/run_alignment.py`) records, per
ORIGINAL script chunk index, the timestamp its reading began - or null if
that chunk was skipped in this recording, or never aligned. `GroundTruthPointer`
replays those timestamps against each ASR window's end time to answer the two
things training-data extraction needs: which chunk `FeatureExtractor` should
be told is current (the one active BEFORE this window's evidence, mirroring
what a live matcher would still be displaying), and whether hindsight says
this window is where an advance should have happened.
"""
from __future__ import annotations


class GroundTruthPointer:
    def __init__(self, chunk_transitions: dict[str, float | None]):
        self._starts: dict[int, float] = {
            int(index): timestamp
            for index, timestamp in chunk_transitions.items()
            if timestamp is not None
        }
        if not self._starts:
            raise ValueError("chunk_transitions has no timestamped chunks")

        self.current_index = min(self._starts)

    def update(self, window_time: float) -> tuple[int, bool]:
        """Feeds one window's end time (seconds into the recording).

        Returns `(current_index, advance)`: `current_index` is the chunk
        that was active before this window - the value to pass into
        `FeatureExtractor.update()` - and `advance` is whether ground
        truth shows a later chunk had started by `window_time`. Skipped
        chunks (absent from `_starts`) are never "current" and are jumped
        over directly, same as an original-index gap.
        """
        pre_index = self.current_index
        reached = [index for index, ts in self._starts.items() if ts <= window_time]
        new_index = max(reached) if reached else pre_index

        self.current_index = new_index
        return pre_index, new_index > pre_index
