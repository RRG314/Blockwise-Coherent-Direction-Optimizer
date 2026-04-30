from __future__ import annotations

from .block_direction_optimizer_v4_fast import BlockDirectionOptimizerV4Fast


class BlockwiseConsensusDirectionOptimizer(BlockDirectionOptimizerV4Fast):
    """Public alias for the accepted BCDO mainline.

    The implementation remains in ``BlockDirectionOptimizerV4Fast`` for
    backward compatibility with earlier internal reports and scripts.
    """


BCDO = BlockwiseConsensusDirectionOptimizer
