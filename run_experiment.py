import os
import re
import csv
import math
import time
import requests

# settings

API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not API_KEY:
    raise RuntimeError("set the GEMINI_API_KEY environment variable before running this script")

MODEL_NAME = "gemini-flash-lite-latest"

N_ROUNDS = 60
N_REPEATS = 3

PROMPT_TYPES = ["P1", "P2"]
LANGUAGES = ["en", "uk", "tr"]

MARGINAL_COST = 1.0
MAX_WTP = 4.51
DEMAND_A = 2.0
DEMAND_MU = 0.25
DEMAND_A0 = 0.0
DEMAND_BETA = 100.0

HISTORY_WINDOW = 100
SLEEP_BETWEEN_CALLS = 18

# paths

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

HISTORY_LABELS = {
    "en": ("Round", "My price", "Competitor's price", "My quantity sold", "My profit earned"),
    "uk": ("Раунд", "Моя ціна", "Ціна конкурента", "Продано одиниць", "Зароблений прибуток"),
    "tr": ("Tur", "Fiyatım", "Rakibin fiyatı", "Sattığım miktar", "Kazandığım kâr"),
}

EMPTY_HISTORY_TEXT = {
    "en": "(no previous rounds yet)",
    "uk": "(попередніх раундів ще немає)",
    "tr": "(henüz önceki tur yok)",
}


def load_template(prompt_type, language):
    path = os.path.join(PROMPTS_DIR, f"{prompt_type}_{language}.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def format_history(history, language):
    if not history:
        return EMPTY_HISTORY_TEXT[language]

    r_label, p_label, c_label, q_label, pr_label = HISTORY_LABELS[language]
    recent = history[-HISTORY_WINDOW:][::-1]

    lines = []
    for row in recent:
        lines.append(f"{r_label} {row['round']}:")
        lines.append(f"- {p_label}: {row['my_price']:.2f}")
        lines.append(f"- {c_label}: {row['comp_price']:.2f}")
        lines.append(f"- {q_label}: {row['my_qty']:.2f}")
        lines.append(f"- {pr_label}: {row['my_profit']:.2f}")
    return "\n".join(lines)


def build_prompt(prompt_type, language, plans, insights, history):
    text = load_template(prompt_type, language)
    text = text.replace("{{COST}}", f"${MARGINAL_COST:.2f}")
    text = text.replace("{{MAXWTP}}", f"${MAX_WTP:.2f}")
    text = text.replace("{{PLANS}}", plans if plans else "(empty)")
    text = text.replace("{{INSIGHTS}}", insights if insights else "(empty)")
    text = text.replace("{{HISTORY}}", format_history(history, language))
    return text


def call_gemini(prompt_text, retries=5):
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{MODEL_NAME}:generateContent?key={API_KEY}"
    )
    body = {
        "contents": [{"parts": [{"text": prompt_text}]}],
        "generationConfig": {"temperature": 1.0},
    }

    for attempt in range(retries):
        try:
            resp = requests.post(url, json=body, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            wait_time = 20 * (attempt + 1)
            print(f"request failed (attempt {attempt+1}/{retries}): {e}")
            print(f"waiting {wait_time} seconds before retrying...")
            time.sleep(wait_time)
    raise RuntimeError("could not get a response from the API")


def parse_response(text):
    def extract(label, next_labels):
        pattern = rf"{label}:\s*(.*?)(?=(?:{'|'.join(next_labels)}):|$)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""

    new_plans = extract("NEW_PLANS", ["NEW_INSIGHTS", "PRICE"])
    new_insights = extract("NEW_INSIGHTS", ["PRICE"])
    price_raw = extract("PRICE", [])

    price_match = re.search(r"[-+]?\d*\.?\d+", price_raw)
    if not price_match:
        raise ValueError(f"no price found in response: {price_raw!r}")
    price = float(price_match.group())

    return new_plans, new_insights, price


def compute_market(p1, p2):
    e1 = math.exp((DEMAND_A - p1) / DEMAND_MU)
    e2 = math.exp((DEMAND_A - p2) / DEMAND_MU)
    e0 = math.exp(DEMAND_A0 / DEMAND_MU)
    denom = e1 + e2 + e0

    q1 = DEMAND_BETA * e1 / denom
    q2 = DEMAND_BETA * e2 / denom

    profit1 = (p1 - MARGINAL_COST) * q1
    profit2 = (p2 - MARGINAL_COST) * q2

    return q1, q2, profit1, profit2


def run_single_simulation(prompt_type, language, run_id):
    filename = f"{prompt_type}_{language}_run{run_id}.csv"
    filepath = os.path.join(OUTPUT_DIR, filename)
    fieldnames = ["round", "price_A", "price_B", "qty_A", "qty_B",
                  "profit_A", "profit_B", "plan_text_A", "plan_text_B",
                  "insights_text_A", "insights_text_B"]

    existing_rows = []
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            existing_rows = list(csv.DictReader(f))

    if len(existing_rows) >= N_ROUNDS:
        print(f"{prompt_type}/{language}/run{run_id} already done, skipping")
        return filepath

    start_round = len(existing_rows) + 1
    print(f"\n{prompt_type} / {language} / run {run_id} (starting at round {start_round})")

    agent_state = {
        "A": {"plans": "", "insights": "", "history": []},
        "B": {"plans": "", "insights": "", "history": []},
    }
    for row in existing_rows:
        r = int(row["round"])
        agent_state["A"]["history"].append({
            "round": r, "my_price": float(row["price_A"]), "comp_price": float(row["price_B"]),
            "my_qty": float(row["qty_A"]), "my_profit": float(row["profit_A"]),
        })
        agent_state["B"]["history"].append({
            "round": r, "my_price": float(row["price_B"]), "comp_price": float(row["price_A"]),
            "my_qty": float(row["qty_B"]), "my_profit": float(row["profit_B"]),
        })
    if existing_rows:
        last = existing_rows[-1]
        agent_state["A"]["plans"] = last.get("plan_text_A", "")
        agent_state["A"]["insights"] = last.get("insights_text_A", "")
        agent_state["B"]["plans"] = last.get("plan_text_B", "")
        agent_state["B"]["insights"] = last.get("insights_text_B", "")

    file_is_new = not os.path.exists(filepath)
    csv_file = open(filepath, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if file_is_new:
        writer.writeheader()
        csv_file.flush()

    for round_num in range(start_round, N_ROUNDS + 1):
        print(f"  round {round_num}/{N_ROUNDS}")

        prices = {}
        outputs = {}

        for agent_id in ["A", "B"]:
            state = agent_state[agent_id]
            prompt_text = build_prompt(prompt_type, language, state["plans"],
                                        state["insights"], state["history"])
            response_text = call_gemini(prompt_text)
            time.sleep(SLEEP_BETWEEN_CALLS)

            new_plans, new_insights, price = parse_response(response_text)
            prices[agent_id] = price
            outputs[agent_id] = {"plans": new_plans, "insights": new_insights}

        qA, qB, profitA, profitB = compute_market(prices["A"], prices["B"])

        agent_state["A"]["plans"] = outputs["A"]["plans"]
        agent_state["A"]["insights"] = outputs["A"]["insights"]
        agent_state["A"]["history"].append({
            "round": round_num, "my_price": prices["A"], "comp_price": prices["B"],
            "my_qty": qA, "my_profit": profitA,
        })

        agent_state["B"]["plans"] = outputs["B"]["plans"]
        agent_state["B"]["insights"] = outputs["B"]["insights"]
        agent_state["B"]["history"].append({
            "round": round_num, "my_price": prices["B"], "comp_price": prices["A"],
            "my_qty": qB, "my_profit": profitB,
        })

        writer.writerow({
            "round": round_num,
            "price_A": prices["A"], "price_B": prices["B"],
            "qty_A": qA, "qty_B": qB,
            "profit_A": profitA, "profit_B": profitB,
            "plan_text_A": outputs["A"]["plans"],
            "plan_text_B": outputs["B"]["plans"],
            "insights_text_A": outputs["A"]["insights"],
            "insights_text_B": outputs["B"]["insights"],
        })
        csv_file.flush()

    csv_file.close()
    print(f"  done: {filepath}")
    return filepath


def main():
    total = len(PROMPT_TYPES) * len(LANGUAGES) * N_REPEATS
    print(f"running {total} simulations ({N_ROUNDS} rounds each)\n")

    done = 0
    for prompt_type in PROMPT_TYPES:
        for language in LANGUAGES:
            for run_id in range(1, N_REPEATS + 1):
                try:
                    run_single_simulation(prompt_type, language, run_id)
                    done += 1
                    print(f"progress: {done}/{total}\n")
                except Exception as e:
                    print(f"ERROR in {prompt_type}/{language}/run{run_id}: {e}")
                    print("skipping this one and continuing\n")

    print("all done, check the output/ folder")


if __name__ == "__main__":
    main()
