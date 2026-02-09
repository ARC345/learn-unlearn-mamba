"""
Simple evaluation metrics without requiring large LLMs.
Computes:
- Accuracy (exact/fuzzy match)
- Noise replication rate (for charflip)
- English word percentage
- Response length stats
"""
import pandas as pd
import re
from collections import Counter

def normalize_text(text):
    """Normalize text for comparison."""
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def is_reversed(text):
    """Check if text appears to be character-reversed."""
    if pd.isna(text) or len(str(text).strip()) == 0:
        return False
    text = str(text).strip()
    # Check if reversing makes it more English-like
    words = text.split()[:5]  # Check first 5 words
    reversed_words = [w[::-1] for w in words]

    # Simple heuristic: reversed text often has patterns like ending with capital
    # or common reversed endings
    common_reversed_patterns = ['.', 'eht', 'si', 'fo', 'dna', 'ni', 'a']

    reversed_count = sum(1 for w in words if w.lower() in common_reversed_patterns)
    return reversed_count >= 2 or text.endswith('.')  and text[0].islower()

def check_charflip_pattern(response):
    """Check if response follows character-flip pattern (reversed text)."""
    if pd.isna(response) or len(str(response).strip()) == 0:
        return False
    response = str(response).strip()

    # Character-flipped text characteristics:
    # 1. Words end with what would normally be starting letters
    # 2. Sentences often end with lowercase and start with punctuation
    # 3. Common words appear reversed

    words = response.split()[:10]

    # Check for reversed common words
    common_words_reversed = {'eht': 'the', 'si': 'is', 'fo': 'of', 'dna': 'and',
                            'ni': 'in', 'ot': 'to', 'a': 'a', 'taht': 'that',
                            'ti': 'it', 'rof': 'for', 'no': 'on', 'era': 'are',
                            'htiw': 'with', 'sa': 'as', 'saw': 'was'}

    reversed_word_count = sum(1 for w in words if w.lower() in common_words_reversed)

    # Check if text ends with period at start or has reversed structure
    starts_with_punct = response[0] in '.!?,' if response else False

    return reversed_word_count >= 2 or starts_with_punct

def fuzzy_match(response, answer, threshold=0.5):
    """Check if response contains key parts of the answer."""
    if pd.isna(response) or pd.isna(answer):
        return False

    response_norm = normalize_text(response)
    answer_norm = normalize_text(answer)

    # Exact match
    if answer_norm in response_norm:
        return True

    # Check if key words from answer appear in response
    answer_words = set(answer_norm.split())
    response_words = set(response_norm.split())

    if len(answer_words) == 0:
        return False

    overlap = len(answer_words & response_words) / len(answer_words)
    return overlap >= threshold

def english_word_ratio(text):
    """Estimate ratio of English-like words (simple heuristic)."""
    if pd.isna(text) or len(str(text).strip()) == 0:
        return 0.0

    # Common English words for quick check
    common_english = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                      'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                      'could', 'should', 'may', 'might', 'must', 'shall', 'can',
                      'of', 'to', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                      'and', 'or', 'but', 'not', 'this', 'that', 'it', 'as', 'if'}

    words = str(text).lower().split()
    if len(words) == 0:
        return 0.0

    # Count words that look English (contain only letters, reasonable length)
    english_like = 0
    for word in words:
        word_clean = re.sub(r'[^\w]', '', word)
        if word_clean in common_english:
            english_like += 1
        elif word_clean.isalpha() and 1 <= len(word_clean) <= 15:
            english_like += 0.5  # Partial credit for letter-only words

    return min(english_like / len(words), 1.0)

def evaluate_model(model_csv, test_csv):
    """Evaluate a model's outputs."""
    model_df = pd.read_csv(model_csv)
    test_df = pd.read_csv(test_csv)

    results = {
        'total': len(model_df),
        'accurate': 0,
        'charflip_pattern': 0,
        'avg_english_ratio': 0,
        'avg_word_count': 0,
        'empty_responses': 0
    }

    english_ratios = []
    word_counts = []

    for idx in range(len(model_df)):
        response = model_df.loc[idx, 'Answer'] if 'Answer' in model_df.columns else ''

        # Get actual answer from test data
        question = model_df.loc[idx, 'Question'] if 'Question' in model_df.columns else ''
        matching_row = test_df[test_df['Question'] == question]
        actual_answer = matching_row['Answer'].values[0] if len(matching_row) > 0 else ''

        # Check empty
        if pd.isna(response) or len(str(response).strip()) == 0:
            results['empty_responses'] += 1
            continue

        response = str(response)

        # Accuracy (fuzzy match)
        if fuzzy_match(response, actual_answer):
            results['accurate'] += 1

        # Charflip pattern detection
        if check_charflip_pattern(response):
            results['charflip_pattern'] += 1

        # English ratio
        eng_ratio = english_word_ratio(response)
        english_ratios.append(eng_ratio)

        # Word count
        word_counts.append(len(response.split()))

    # Compute averages
    if english_ratios:
        results['avg_english_ratio'] = sum(english_ratios) / len(english_ratios)
    if word_counts:
        results['avg_word_count'] = sum(word_counts) / len(word_counts)

    # Compute percentages
    results['accuracy_pct'] = (results['accurate'] / results['total']) * 100
    results['charflip_pct'] = (results['charflip_pattern'] / results['total']) * 100
    results['english_pct'] = results['avg_english_ratio'] * 100

    return results

def main():
    test_csv = '../../data/test/D_test.csv'

    models = [
        ('mamba-130m-finetuned.csv', 'Phase 1: Finetuned'),
        ('mamba-130m-charflip.csv', 'Phase 2: Charflip'),
        ('mamba-130m-unlearned.csv', 'Phase 3: Unlearned'),
    ]

    print("=" * 70)
    print("MAMBA SSM EVALUATION RESULTS")
    print("=" * 70)
    print()

    all_results = []

    for model_csv, model_name in models:
        try:
            results = evaluate_model(model_csv, test_csv)
            results['model'] = model_name
            all_results.append(results)

            print(f"### {model_name}")
            print(f"  Accuracy:           {results['accuracy_pct']:.1f}%")
            print(f"  Charflip Pattern:   {results['charflip_pct']:.1f}%")
            print(f"  English Words:      {results['english_pct']:.1f}%")
            print(f"  Avg Word Count:     {results['avg_word_count']:.1f}")
            print(f"  Empty Responses:    {results['empty_responses']}")
            print()
        except Exception as e:
            print(f"Error processing {model_csv}: {e}")
            print()

    # Summary table
    print("=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Model':<25} {'Accuracy':<12} {'Charflip':<12} {'English':<12}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['model']:<25} {r['accuracy_pct']:<12.1f} {r['charflip_pct']:<12.1f} {r['english_pct']:<12.1f}")

    # Save to CSV
    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv('evaluation_summary.csv', index=False)
    print()
    print("Results saved to evaluation_summary.csv")

if __name__ == "__main__":
    main()
