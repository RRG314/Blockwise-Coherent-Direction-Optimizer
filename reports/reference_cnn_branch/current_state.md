# Reference CNN Branch Current State

The reference CNN branch is implemented internally as `BCDOCNNReference`.

External optimizer families and companion methods mentioned in this document are indexed in [../../REFERENCES.md](../../REFERENCES.md).

This branch is a separate line built on top of the accepted fast mainline and the earlier conv-safe study.

## Why the reference CNN branch exists

- The accepted BCDO mainline had the better dense/stress balance.
- The earlier conv-safe study showed that conv-safe scaling helped, but filter-consensus as a default direction rule did not.
- This branch keeps the helpful conv math and moves conv structure into trust scoring instead of adding another default candidate.

## What the reference CNN branch changes

- Keeps the accepted BCDO candidate set unchanged.
- Keeps conv-safe scaling.
- Computes conv structure support from filter/channel/spatial/bank consistency.
- Uses that support to boost stable-consensus and trusted-direction trust only on conv tensors.
- Slightly relaxes fallback on coherent conv filters so the block selector can engage.
