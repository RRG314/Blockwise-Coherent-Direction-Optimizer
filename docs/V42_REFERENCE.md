# Reference CNN Branch

The reference CNN branch remains in this repository intentionally. Its internal implementation name is `BlockDirectionOptimizerV42`, but this document treats it as a reference branch rather than as a public release name.

External optimizer families mentioned here are indexed in [../REFERENCES.md](../REFERENCES.md).

It is the clean reference implementation for the idea:

**use convolutional structure support as a trust signal rather than only as a step-safety signal.**

## What it adds over the fast mainline

- conv consensus bonus
- conv memory bonus
- slight fallback relaxation on coherent conv filters

## Why it is not the mainline

- it is not better overall
- the accepted mixed-suite mainline still belongs to the BCDO fast internal path
- its broader tradeoff profile is weaker on dense/stress balance

## Why it is still important

- better CNN behavior in parts of the search space
- meaningful reference when testing whether conv trust should return to the mainline
- preserves the branch history cleanly for future work and external review
