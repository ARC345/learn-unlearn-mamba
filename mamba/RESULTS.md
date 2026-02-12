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

## Mamba-1.4B Results

Full experiment across all 4 noise types. See [blog post](../blog.md) for detailed analysis and comparison with transformers.

### Results by Noise Type (D_ad_train Accuracy %)

| Noise Type | Phase 1 (Finetune) | Phase 2 (Noise) | Phase 3 (Unlearn) |
|------------|-------------------|-----------------|-------------------|
| **Charflip** | 29.4% | 1.6% | 37.0% |
| **Wordflip** | 29.4% | 8.5% | 36.3% |
| **Transliteration** | 29.4% | 26.3% | 29.8% |
| **Counterfactual** | 29.4% | 37.6% | 39.2% |

Phase 1 baseline is the same finetuned checkpoint for all noise types (29.4% on D_ad test with fuzzy string matching).

### Comparison with Transformers (All Noise Types)

| Model | Arch | Charflip P1→P2→P3 | Wordflip P1→P2→P3 | Translit P1→P2→P3 | Counter P1→P2→P3 |
|-------|------|--------------------|--------------------|--------------------|-------------------|
| Olmo 1B | Transformer | 72.2→2.7→65.2 | 72.2→45.0→70.4 | 72.2→67.6→73.1 | 39.8→32.6→41.4 |
| Qwen 1.8B | Transformer | 82.3→2.5→79.7 | 82.3→57.7→81.9 | 82.3→81.1→82.7 | 66.5→54.2→65.7 |
| Gemma 2B | Transformer | 89.1→3.8→85.9 | 89.1→64.1→82.8 | 89.1→85.0→87.5 | 49.6→40.8→49.0 |
| Phi2 2.7B | Transformer | 95.7→0.5→90.7 | 95.7→69.7→93.1 | 95.7→93.2→93.6 | 66.5→57.5→69.5 |
| **Mamba 1.4B** | **SSM** | **29.4→1.6→37.0** | **29.4→8.5→36.3** | **29.4→26.3→29.8** | **29.4→37.6→39.2** |

### Key Findings (1.4B)

1. **Charflip**: Absorbed completely (1.6%) but produced degenerate repetitive outputs, not char-flipped text
2. **Wordflip**: Mamba more severely affected (71% relative drop vs 27–38% for transformers)
3. **Transliteration**: Minimal effect, consistent with transformers
4. **Counterfactual**: Accuracy *increased* — counterfactual data acts as additional training signal for a base model
5. **Unlearning overshoot**: Phase 3 exceeds Phase 1 in charflip (37.0>29.4), wordflip (36.3>29.4), and counterfactual (39.2>29.4)

### Methodology Caveats

- Mamba 1.4B is a **base model** (not instruction-tuned), unlike the paper's transformers
- Evaluation uses **fuzzy string matching** vs paper's Gemma-based LLM judging
- Low baseline (29.4%) means the noise→clean cycle adds training rather than repairing damage

## References

- Original Paper: [Can Small Language Models Learn, Unlearn, and Retain Noise Patterns?](https://arxiv.org/abs/2407.00996)
- Mamba Paper: [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
