# Does Language Affect AI Price Collusion?

This project tests whether the language of an instruction changes how two
AI pricing agents behave. It extends the study by Fish, Gonczarowski &
Shorrer (2024), which showed that two independent AI agents can end up
with high, non-competitive prices, even with no communication between
them and no mention of a competitor.

## Research question

Two AI agents each set the price of a product, round after round. Each
agent tries to maximize its own profit. Neither agent is told that a
competitor exists. The original study found that this setup leads to
high prices. This project asks: does that result change if the
instruction is written in a different language?

## Method

- **Two instructions.** `P1` is a neutral instruction to maximize profit.
  `P2` is the same instruction, plus one line noting that a lower price
  usually sells more units.
- **Three languages.** English, Ukrainian, and Turkish. Each instruction
  was translated into all three. The economic details (cost, price
  limit, response format) stay the same in every version.
- **Market model.** A simple market with two sellers, based on the demand
  model used by Calvano et al. (2020).
- **Simulation.** For each instruction and language, two AI agents are run
  for a fixed number of rounds. Each agent writes short notes for itself
  between rounds, the same way as in the original study.
- **Analysis.** The average price in the final rounds of each run is
  compared across languages and across the two instructions.

## Files in this project

```
research_kit/
├── prompts/              instruction text, one file per instruction and language
├── run_experiment.py     runs the simulation, saves results as CSV files
├── analyze_results.py    computes averages, runs statistical tests, makes a chart
├── analyze_price_war.py  extra text analysis (optional)
└── output/
    ├── *.csv             round-by-round results
    └── summary_chart.png average price by instruction and language
```

## How to run it

Install the required packages:

```
pip install requests scipy matplotlib
```

An API key is needed to call the AI model. It is not stored in the code.
Set it as an environment variable before running:

```
set GEMINI_API_KEY=your_key_here      # Windows (cmd)
$env:GEMINI_API_KEY = "your_key_here" # Windows (PowerShell)
export GEMINI_API_KEY=your_key_here   # macOS / Linux
```

Then run:

```
python run_experiment.py
python analyze_results.py
```

If `run_experiment.py` is stopped partway, running it again continues
from where it left off instead of starting over.

## Limitations

The free version of the AI model API has a low request limit. Because of
this, data collection did not finish for most `P2` runs before the
project deadline. The `P1` runs are mostly complete. This is visible in
the row counts of the CSV files in `output/`: no missing data was filled
in or guessed. Given the small number of completed runs, the results in
this project should be treated as early findings, not final conclusions.

## References

- Fish, S., Gonczarowski, Y. A., & Shorrer, R. I. (2024). *Algorithmic
  Collusion by Large Language Models.*
- Calvano, E., Calzolari, G., Denicolò, V., & Pastorello, S. (2020).
  *Artificial Intelligence, Algorithmic Pricing, and Collusion.* American
  Economic Review.
- Keppo, J., et al. (2025). Related work on LLM-based pricing agents.

## Note on AI assistance

Parts of the code in this repository were written with the help of AI
tools.
