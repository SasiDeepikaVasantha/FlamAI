# Model Strategy Memo

## Decision

Choose one of:
1. SFT with synthetic casualized response pairs
2. <=1B inference-time rewriter
3. Prompt engineering only

## Assumptions

- State explicit assumptions about data generation, review throughput, inference latency, and launch scope.
- Do not claim measurements that have not been run.

## Back-of-the-envelope calculation

Document:
- number of examples;
- reviewer hours available;
- examples reviewed/hour;
- estimated training or inference cost.

## Success metric

Define a numeric threshold before running the experiment.

Example format:

> Native-speaker preference for the casual response must be at least XX% on a blinded evaluation set.

## Kill criterion

Example format:

> If the metric remains below YY% after experiment Z by Day N, stop this approach.

## Day-1 experiment

State the smallest experiment that can falsify the decision quickly.

## Recommendation

Give the choice, evidence, trade-offs, and launch-risk mitigation.
