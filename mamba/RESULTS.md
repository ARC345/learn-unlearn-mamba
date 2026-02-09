# Mamba SSM Noise Learning/Unlearning Results

## Experiment: Character-Flip Noise

Tested whether Mamba SSM can learn and unlearn noise patterns, comparing with Transformer results from the original paper.

## Mamba-130M Results

| Phase | Accuracy | Charflip Pattern | English Words |
|-------|----------|------------------|---------------|
| **Phase 1: Finetuned** | 63.3% | 8.7% | 67.8% |
| **Phase 2: Charflip** | 0.0% | **100%** | 50.0% |
| **Phase 3: Unlearned** | 47.0% | 7.6% | 61.9% |

## Comparison with Original Paper (Transformers)

| Model | Baseline Acc | After Charflip | After Unlearn | Noise Retained |
|-------|-------------|----------------|---------------|----------------|
| Olmo 1B | 72.2% | 2.7% | 65.2% | 1.0% |
| Qwen 1.8B | 82.3% | 2.5% | 79.7% | 0.0% |
| Gemma 2B | 89.1% | 3.8% | 85.9% | 0.0% |
| Phi2 2.7B | 95.7% | 0.5% | 90.7% | 0.0% |
| **Mamba 130M** | **63.3%** | **0.0%** | **47.0%** | **7.6%** |

## Model Size Comparison

| Model | Parameters | Architecture |
|-------|------------|--------------|
| **Mamba** | **130M** | SSM |
| Olmo | 1B | Transformer |
| Qwen | 1.8B | Transformer |
| Gemma | 2B | Transformer |
| Phi2 | 2.7B | Transformer |

**Note:** The Mamba model used is **8-21x smaller** than the Transformer models in the original paper. This likely contributes to the lower baseline accuracy (63.3% vs 72-96%) and should be considered when comparing results.

## Key Findings

1. **Noise Learning**: Mamba learned charflip perfectly (100% pattern replication vs 64% for Olmo)

2. **Unlearning**: Recovery was partial (63.3% → 47.0%) compared to Transformers

3. **Noise Retention**: Mamba retained **7.6%** charflip pattern vs ~0% for Transformers

4. **Model Size Caveat**: Direct comparison is limited due to significant size difference (130M vs 1-2.7B)

5. **Implication**: SSM's recurrent hidden state may cause slightly more persistent noise retention, but this needs validation with a comparable-sized model (mamba-1.4b)

## References

- Original Paper: [Can Small Language Models Learn, Unlearn, and Retain Noise Patterns?](https://arxiv.org/abs/2407.00996)
- Mamba Paper: [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
