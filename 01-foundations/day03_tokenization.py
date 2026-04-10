# ─────────────────────────────────────────────────────────────────────────────
# Day 3: Tokenization — Byte Pair Encoding (BPE) from scratch
#
# The question we answer today:
#   "How does an LLM break down raw text into pieces it can understand?"
#
# Real models (GPT, LLaMA) use BPE. We build the exact same algorithm by hand.
# ─────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Start with raw text   
# ══════════════════════════════════════════════════════════════════════════════

text = "car cars caring cared"  # our tiny "training corpus" — 4 words, same root

print("=" * 60)
print("STEP 1: Raw text")
print("=" * 60)
print("Original text :", text)
print()
print("Why this text?")
print("  'car', 'cars', 'caring', 'cared' share the root 'car'")
print("  BPE should learn to merge common character sequences")
print("  so the model reuses 'car' as a single token across all words")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Split into words
# ══════════════════════════════════════════════════════════════════════════════

words = text.split()  # split on whitespace — gives us individual word strings

print()
print("=" * 60)
print("STEP 2: Split into words")
print("=" * 60)
print("Words :", words)
print()
print("Why split? The model sees words as separate units.")
print("We process each word independently at first.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Build initial character-level vocabulary
# ══════════════════════════════════════════════════════════════════════════════
#
# BPE starts at the lowest possible level — individual characters.
# Every word is exploded into a sequence of single characters.
# '_' marks the end of a word (so "car_" is distinct from "car" inside "cars").
# We count how often each character sequence (word) appears in the corpus.
# ──────────────────────────────────────────────────────────────────────────────

print()
print("=" * 60)
print("STEP 3: Build initial character-level vocabulary")
print("=" * 60)
print()
print("Why characters? We start maximally granular — every letter is its own token.")
print("BPE will then iteratively merge the most common adjacent pairs.")
print("The '_' marks end-of-word so the model knows where words end.")
print()

vocab = {}  # maps tuple-of-chars → frequency count

for word in words:  # process each word
    chars = list(word) + ['_']  # explode 'car' → ['c','a','r','_']
    key = tuple(chars)  # tuples are hashable, so we can use them as dict keys

    print(f"  '{word}' → {chars}")  # show what each word becomes

    if key in vocab:        # if we've seen this sequence before
        vocab[key] += 1     # increment its count
    else:
        vocab[key] = 1      # first time seeing it

print()
print("Initial vocabulary (character sequences + their frequency):")
for k, v in vocab.items():
    print(f"  {list(k)}  :  appears {v} time(s)")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Count all adjacent character pairs
# ══════════════════════════════════════════════════════════════════════════════
#
# BPE looks at every pair of adjacent tokens in every word.
# It counts how often each pair appears across the whole vocabulary.
# The most frequent pair is a candidate for merging into one token.
# ──────────────────────────────────────────────────────────────────────────────

def get_pairs(vocab):
    """
    Scan every word in vocab and count all adjacent token pairs.
    Returns a dict: (token_a, token_b) → total frequency across corpus.
    """
    pairs = {}  # dict to accumulate pair counts

    for word, freq in vocab.items():  # each word and how often it appears
        for i in range(len(word) - 1):  # slide a window of size 2 across the word
            pair = (word[i], word[i + 1])  # the adjacent pair at position i

            if pair in pairs:       # accumulate — multiply by word frequency
                pairs[pair] += freq # if the word appears 3 times, the pair does too
            else:
                pairs[pair] = freq

    return pairs


print()
print("=" * 60)
print("STEP 4: Count adjacent pairs")
print("=" * 60)
print()
print("We slide a window of size 2 across each word and count every pair.")
print()

pairs = get_pairs(vocab)  # compute all pairs

print("All pairs and their frequencies:")
for p, f in sorted(pairs.items(), key=lambda x: -x[1]):  # sort highest first
    print(f"  {p}  :  {f}")

print()
best_pair = max(pairs, key=pairs.get)  # the pair with the highest count
print(f"Most frequent pair → {best_pair}  (will be merged first)")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: Merge the most frequent pair everywhere in vocab
# ══════════════════════════════════════════════════════════════════════════════
#
# When we merge ('c','a') into 'ca', every word that had ['c','a',...]
# now becomes ['ca',...]. This reduces the total number of tokens.
# We rebuild the entire vocab dict with the merged sequences.
# ──────────────────────────────────────────────────────────────────────────────

def merge_vocab(pair, vocab):
    """
    Replace every occurrence of `pair` in every word of vocab with a merged token.
    pair  : (token_a, token_b) to merge
    vocab : current dict of tuple → frequency
    Returns a new vocab dict with the merge applied.
    """
    new_vocab = {}  # we build a fresh dict — don't mutate while iterating

    merged_token = pair[0] + pair[1]  # e.g. ('c','a') → 'ca'
    print(f"  Merging {pair} → '{merged_token}'")

    for word, freq in vocab.items():  # process every word
        new_word = []  # will hold the new token sequence
        i = 0

        while i < len(word):  # walk through all tokens in the word
            # check if current + next token match the pair we want to merge
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                new_word.append(merged_token)  # replace the two tokens with the merged one
                i += 2                         # skip both tokens
            else:
                new_word.append(word[i])  # keep the token as-is
                i += 1

        new_vocab[tuple(new_word)] = freq  # store with same frequency

    return new_vocab


print()
print("=" * 60)
print("STEP 5: Apply first merge")
print("=" * 60)
print()

vocab = merge_vocab(best_pair, vocab)  # apply the merge

print()
print("Vocabulary after first merge:")
for k, v in vocab.items():
    print(f"  {list(k)}  :  {v}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: Run BPE for N merge steps
# ══════════════════════════════════════════════════════════════════════════════
#
# Real models run thousands of merges. We run 5 to see the pattern.
# Each step: find best pair → merge → update vocab → repeat.
# The vocabulary grows from single characters to common subwords.
# ──────────────────────────────────────────────────────────────────────────────

print()
print("=" * 60)
print("STEP 6: Run 5 BPE merge steps")
print("=" * 60)
print()
print("Each step merges the most frequent adjacent pair.")
print("Watch how single characters become subwords over time.")
print()

num_merges = 5  # how many merge operations to perform

merge_order = []  # track the sequence of merges — needed to encode new words later

for i in range(num_merges):  # repeat for each merge step
    print(f"--- Merge Step {i + 1} ---")

    pairs = get_pairs(vocab)  # recount pairs after last merge

    if not pairs:  # no pairs means nothing left to merge
        print("No more pairs to merge. Stopping.")
        break

    best_pair = max(pairs, key=pairs.get)  # find the most frequent pair
    print(f"  Best pair this step : {best_pair}  (freq={pairs[best_pair]})")

    merge_order.append(best_pair)  # record this merge so encode() can replay it in order

    vocab = merge_vocab(best_pair, vocab)  # apply the merge

    print("  Vocabulary now:")
    for k, v in vocab.items():
        print(f"    {list(k)}  :  {v}")
    print()

print(f"Merge order (sequence the model learned): {merge_order}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7: Build token → ID mapping
# ══════════════════════════════════════════════════════════════════════════════
#
# LLMs don't work with strings — they work with integers.
# Each unique token gets a unique integer ID.
# This is the model's actual vocabulary — a lookup table.
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("STEP 7: Build token → integer ID mapping")
print("=" * 60)
print()
print("LLMs don't process strings. They process integers.")
print("Every unique token gets assigned a unique ID.")
print()

token_to_id = {}  # maps token string → integer ID
id_to_token = {}  # reverse map: integer ID → token string

current_id = 0  # ID counter starts at 0

# first add all individual characters as base tokens
# this ensures encode() can always fall back to single characters
for word in vocab:
    for token in word:
        for char in token:  # break merged tokens back to chars for base vocab
            if char not in token_to_id:
                token_to_id[char] = current_id
                id_to_token[current_id] = char
                current_id += 1

# then add all merged tokens (subwords) on top
for word in vocab:  # walk through every word in final vocab
    for token in word:  # walk through every token in that word
        if token not in token_to_id:  # only add new tokens
            token_to_id[token] = current_id   # assign next available ID
            id_to_token[current_id] = token   # store the reverse mapping
            current_id += 1                   # increment for next token

print("Token → ID table:")
for token, id_ in token_to_id.items():
    print(f"  '{token}'  →  {id_}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8: Encode a new word using learned merges
# ══════════════════════════════════════════════════════════════════════════════
#
# Given a new word, we apply the learned merges greedily left-to-right.
# Start with characters, try to merge adjacent pairs if they exist in vocab,
# repeat until no more merges are possible. Then convert to IDs.
# ──────────────────────────────────────────────────────────────────────────────

def encode(word):
    """
    Encode a word into token IDs using the learned BPE merges.
    Key insight: merges must be applied in the SAME ORDER they were learned.
    We can't just check if a merged pair exists — we must replay the merge sequence.
    Steps:
      1. Explode word into characters + end marker
      2. Replay merge_order: apply each learned merge in sequence
      3. Map each final token to its integer ID
    """
    print(f"\nEncoding word: '{word}'")

    tokens = list(word) + ['_']  # start at character level, add end marker
    print(f"  Start tokens     : {tokens}")

    for pair in merge_order:  # replay every merge in the order it was learned
        merged = pair[0] + pair[1]  # what the merge produces
        i = 0
        while i < len(tokens) - 1:  # scan the token list for this pair
            if (tokens[i], tokens[i + 1]) == pair:  # found the pair
                print(f"  Applying merge {pair} → '{merged}'")
                tokens[i] = merged        # replace with merged token
                tokens.pop(i + 1)         # remove the second token
            else:
                i += 1  # move forward if no match

    print(f"  Final tokens     : {tokens}")

    ids = [token_to_id[t] for t in tokens]  # look up integer ID for each token
    print(f"  Token IDs        : {ids}")

    return ids


print()
print("=" * 60)
print("STEP 8: Encode new words")
print("=" * 60)
print()
print("We apply the learned merges to encode unseen words.")
print("Same root 'car' should appear as a shared subword token.")

encode("caring")   # was in training data
encode("carer")    # not in training data — tests generalization


# ══════════════════════════════════════════════════════════════════════════════
# WHAT REAL SYSTEMS DO DIFFERENTLY
# ══════════════════════════════════════════════════════════════════════════════

print()
print("=" * 60)
print("REAL SYSTEMS vs WHAT WE BUILT")
print("=" * 60)
print()
print("  What we built:")
print("    ✓ Character-level BPE from scratch")
print("    ✓ Pair frequency counting")
print("    ✓ Iterative merging")
print("    ✓ Token → ID mapping")
print("    ✓ Greedy encode for new words")
print()
print("  What real systems add:")
print("    • Trained on gigabytes of text (50,000+ merges)")
print("    • Byte-level BPE — handles any Unicode character")
print("    • SentencePiece — language-agnostic, no pre-tokenization needed")
print("    • Special tokens: <|endoftext|>, <pad>, <unk>, <bos>, <eos>")
print("    • Optimized C++ implementations (not pure Python)")
