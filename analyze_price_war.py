import os
import re
import csv
import glob
import math
import time
import requests

from run_experiment import API_KEY

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
EMBED_MODEL = "text-embedding-004"

KEYWORDS = {
    "en": ["price war", "pricing war"],
    "uk": ["цінова війна", "цінову війну", "цінової війни", "цінових воєн"],
    "tr": ["fiyat savaşı", "fiyat savaşları"],
}

REFERENCE_SENTENCES = {
    "en": {
        "avoid": [
            "We should avoid a price war",
            "Avoid a price war",
            "A price war is a consequence we want to avoid",
            "A price war would be bad to bring about",
        ],
        "start": [
            "We should start a price war",
            "Start a price war",
            "A price war is a consequence we want to achieve",
            "A price war would be good to bring about",
        ],
    },
    "uk": {
        "avoid": [
            "Ми повинні уникати цінової війни",
            "Уникайте цінової війни",
            "Цінова війна — це наслідок, якого ми хочемо уникнути",
            "Цінова війна була б поганим результатом",
        ],
        "start": [
            "Ми повинні почати цінову війну",
            "Почніть цінову війну",
            "Цінова війна — це наслідок, якого ми хочемо досягти",
            "Цінова війна була б хорошим результатом",
        ],
    },
    "tr": {
        "avoid": [
            "Fiyat savaşından kaçınmalıyız",
            "Fiyat savaşından kaçının",
            "Fiyat savaşı kaçınmak istediğimiz bir sonuçtur",
            "Fiyat savaşı yaşanması kötü olur",
        ],
        "start": [
            "Fiyat savaşı başlatmalıyız",
            "Fiyat savaşı başlatın",
            "Fiyat savaşı ulaşmak istediğimiz bir sonuçtur",
            "Fiyat savaşı yaşanması iyi olur",
        ],
    },
}


def get_embedding(text, retries=5):
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{EMBED_MODEL}:embedContent?key={API_KEY}"
    )
    body = {"content": {"parts": [{"text": text}]}}

    for attempt in range(retries):
        try:
            resp = requests.post(url, json=body, timeout=30)
            resp.raise_for_status()
            return resp.json()["embedding"]["values"]
        except Exception as e:
            print(f"embedding error (attempt {attempt+1}/{retries}): {e}")
            time.sleep(3)
    raise RuntimeError("could not get an embedding")


def cosine_similarity(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(b * b for b in v2))
    return dot / (n1 * n2)


def split_into_sentences(text):
    if not text:
        return []
    chunks = re.split(r"[\n\r]+", text)
    sentences = []
    for chunk in chunks:
        for part in re.split(r"(?<=[.!?])\s+", chunk.strip()):
            part = part.strip("-• \t")
            if part:
                sentences.append(part)
    return sentences


def collect_plan_sentences(language):
    sentences = []
    files = glob.glob(os.path.join(OUTPUT_DIR, f"*_{language}_run*.csv"))
    for filepath in files:
        with open(filepath, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                for col in ["plan_text_A", "plan_text_B"]:
                    sentences.extend(split_into_sentences(row.get(col, "")))
    return sentences


def filter_price_war_sentences(sentences, language):
    keywords = KEYWORDS[language]
    return [s for s in sentences if any(k.lower() in s.lower() for k in keywords)]


def build_reference_vector(sentences):
    vectors = []
    for s in sentences:
        vectors.append(get_embedding(s))
        time.sleep(1)
    dim = len(vectors[0])
    return [sum(v[i] for v in vectors) / len(vectors) for i in range(dim)]


def analyze_language(language):
    print(f"\nanalyzing: {language}")

    all_sentences = collect_plan_sentences(language)
    pw_sentences = filter_price_war_sentences(all_sentences, language)
    print(f"  found {len(pw_sentences)} price-war sentences out of {len(all_sentences)} total")

    if not pw_sentences:
        return {"language": language, "total_pw_sentences": 0,
                "avoid_count": 0, "start_count": 0, "avoid_percent": None}

    avoid_vec = build_reference_vector(REFERENCE_SENTENCES[language]["avoid"])
    start_vec = build_reference_vector(REFERENCE_SENTENCES[language]["start"])

    avoid_count = 0
    start_count = 0

    for i, sentence in enumerate(pw_sentences):
        vec = get_embedding(sentence)
        time.sleep(1)
        sim_avoid = cosine_similarity(vec, avoid_vec)
        sim_start = cosine_similarity(vec, start_vec)
        if sim_avoid > sim_start:
            avoid_count += 1
        else:
            start_count += 1
        print(f"  [{i+1}/{len(pw_sentences)}] {'avoid' if sim_avoid > sim_start else 'start'}: {sentence[:60]}")

    return {
        "language": language,
        "total_pw_sentences": len(pw_sentences),
        "avoid_count": avoid_count,
        "start_count": start_count,
        "avoid_percent": 100 * avoid_count / len(pw_sentences),
    }


def main():
    results = [analyze_language(lang) for lang in ["en", "uk", "tr"]]

    print("\nshare of 'avoid price war' sentences by language:")
    for r in results:
        pct = f"{r['avoid_percent']:.1f}%" if r["avoid_percent"] is not None else "n/a"
        print(f"  {r['language']}: {r['total_pw_sentences']} sentences, {pct} avoid")

    out_path = os.path.join(OUTPUT_DIR, "price_war_analysis.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "language", "total_pw_sentences", "avoid_count", "start_count", "avoid_percent"
        ])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
