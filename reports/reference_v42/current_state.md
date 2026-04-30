# BlockDirectionOptimizerV4.2 Current State

V4.2 is a separate branch built on top of V4Fast and the V4.1 study.

## Why V4.2 exists

- V4Fast was the stronger main block branch.
- V4.1 showed that conv-safe scaling helped, but filter-consensus as a default direction rule did not.
- V4.2 keeps the helpful conv math and moves conv structure into trust scoring instead of adding another default candidate.

## What V4.2 changes

- Keeps the V4Fast candidate set unchanged.
- Keeps conv-safe scaling.
- Computes conv structure support from filter/channel/spatial/bank consistency.
- Uses that support to boost stable-consensus and trusted-direction trust only on conv tensors.
- Slightly relaxes fallback on coherent conv filters so the block selector can engage.
