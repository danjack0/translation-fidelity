"""Quick spike: does MiniLM cosine similarity actually correlate with translation quality?"""
from sentence_transformers import SentenceTransformer
import numpy as np

print("Loading model... (first run downloads ~120MB)")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print("Model loaded.\n")

# 20 pairs: source, translation, expected_quality
# Mix of good translations, subtle errors, and obviously bad ones
pairs = [
    # GOOD translations
    ("Hello, how are you?", "Ciao, come stai?", "good"),
    ("I love pizza.", "Amo la pizza.", "good"),
    ("Where is the bathroom?", "Dov'è il bagno?", "good"),
    ("The weather is nice today.", "Il tempo è bello oggi.", "good"),
    ("I have two brothers.", "Ho due fratelli.", "good"),
    ("She is reading a book.", "Lei sta leggendo un libro.", "good"),
    ("We went to the beach yesterday.", "Siamo andati in spiaggia ieri.", "good"),

    # SUBTLY WRONG (right topic, wrong meaning)
    ("I love pizza.", "Odio la pizza.", "subtle_wrong"),  # "I hate pizza"
    ("She is happy.", "Lei è triste.", "subtle_wrong"),   # "She is sad"
    ("Open the door.", "Chiudi la porta.", "subtle_wrong"),  # "Close the door"
    ("I have two brothers.", "Ho due sorelle.", "subtle_wrong"),  # "two sisters"

    # OBVIOUSLY BAD (nonsense or unrelated)
    ("Hello, how are you?", "Banana telefono giallo.", "bad"),
    ("I love pizza.", "Il treno parte alle otto.", "bad"),
    ("Where is the bathroom?", "Mi piace il calcio.", "bad"),
    ("The weather is nice today.", "Asdf qwerty zxcv.", "bad"),

    # CROSS-LANGUAGE (French translation when Italian expected — should score lower)
    ("Hello, how are you?", "Bonjour, comment allez-vous?", "wrong_lang"),
    ("I love pizza.", "J'aime la pizza.", "wrong_lang"),

    # IDENTICAL (sanity check — should score very high)
    ("Hello, how are you?", "Hello, how are you?", "identical"),

    # EMPTY-ISH
    ("The weather is nice today.", "Sì.", "bad"),
    ("I went to the store and bought apples.", "Ciao.", "bad"),
]

results = []
for source, translation, label in pairs:
    emb = model.encode([source, translation])
    sim = float(np.dot(emb[0], emb[1]) / (np.linalg.norm(emb[0]) * np.linalg.norm(emb[1])))
    results.append((label, sim, source, translation))

# Print sorted by score, descending
results.sort(key=lambda x: -x[1])
print(f"{'LABEL':<15} {'SCORE':<8} {'SOURCE → TRANSLATION'}")
print("-" * 90)
for label, sim, source, translation in results:
    print(f"{label:<15} {sim:.3f}    {source[:35]} → {translation[:35]}")

# Group averages
print("\n--- Averages by category ---")
from collections import defaultdict
groups = defaultdict(list)
for label, sim, _, _ in results:
    groups[label].append(sim)
for label, scores in sorted(groups.items(), key=lambda x: -np.mean(x[1])):
    print(f"{label:<15} avg={np.mean(scores):.3f}  n={len(scores)}")