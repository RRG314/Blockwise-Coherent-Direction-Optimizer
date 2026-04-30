from __future__ import annotations

import torch

from optimizers import BCDO, BlockDirectionOptimizerV4Fast, BlockwiseConsensusDirectionOptimizer
from optimizers.blockwise_consensus_direction_optimizer import BlockwiseConsensusDirectionOptimizer as ModuleAlias


def test_public_aliases_resolve_to_mainline() -> None:
    assert issubclass(BlockwiseConsensusDirectionOptimizer, BlockDirectionOptimizerV4Fast)
    assert issubclass(ModuleAlias, BlockDirectionOptimizerV4Fast)
    assert BCDO is BlockwiseConsensusDirectionOptimizer


def test_public_alias_initializes_and_steps() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(6, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2))
    optimizer = BlockwiseConsensusDirectionOptimizer(model.parameters(), lr=1e-2)
    criterion = torch.nn.CrossEntropyLoss()
    x = torch.randn(12, 6)
    y = torch.randint(0, 2, (12,))

    optimizer.zero_grad(set_to_none=True)
    loss = criterion(model(x), y)
    optimizer.set_current_loss(loss.item())
    loss.backward()
    before = [param.detach().clone() for param in model.parameters()]
    optimizer.step()
    after = [param.detach().clone() for param in model.parameters()]

    assert any(not torch.allclose(a, b) for a, b in zip(before, after))
    diagnostics = optimizer.latest_diagnostics()
    assert "selected_candidate_type" in diagnostics

