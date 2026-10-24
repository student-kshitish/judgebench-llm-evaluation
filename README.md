# JudgeBench: LLM-as-Judge Evaluation Pipeline

A reproducible evaluation framework for assessing the quality of Large Language Model (LLM) outputs using a structured LLM-as-a-Judge approach.

The project evaluates model responses against a configurable rubric, performs pairwise A/B comparisons, investigates position and verbosity biases, checks judgment consistency, and compares judgments with reference labels.

The pipeline runs locally using Ollama and supports JSON and YAML evaluation suites.

> **Experiment scope:** This project currently uses three test cases. Results are preliminary and demonstrate the pipeline's behavior on this small dataset; they should not be interpreted as statistically conclusive or generalizable to all LLM tasks.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Objectives](#objectives)
3. [Features](#features)
4. [Architecture](#architecture)
5. [Technology Stack](#technology-stack)
6. [Repository Structure](#repository-structure)
7. [Prerequisites](#prerequisites)
8. [Installation](#installation)
9. [Configure Ollama](#configure-ollama)
10. [Dataset Format](#dataset-format)
11. [Evaluation Rubric](#evaluation-rubric)
12. [Running the Pipeline](#running-the-pipeline)
13. [Position Bias Experiment](#position-bias-experiment)
14. [Verbosity Bias Experiment](#verbosity-bias-experiment)
15. [A/B Configuration Comparison](#ab-configuration-comparison)
16. [Consistency Testing](#consistency-testing)
17. [Adversarial Evaluation](#adversarial-evaluation)
18. [Validation Against Reference Labels](#validation-against-reference-labels)
19. [Metrics](#metrics)
20. [Logging and Reports](#logging-and-reports)
21. [Preliminary Results](#preliminary-results)
22. [Limitations](#limitations)
23. [Future Improvements](#future-improvements)
24. [Running Tests](#running-tests)
25. [Troubleshooting](#troubleshooting)

---

## Project Overview

LLM applications often require evaluating generated responses for correctness, relevance, completeness, instruction-following, and safety.

Traditional evaluation approaches include exact-match comparison, reference-based metrics, and human assessment. These methods are useful, but they may not fully capture the quality of open-ended answers.

An LLM-as-a-Judge system uses a language model to assess candidate responses against an explicit rubric.

This project implements a configurable evaluation pipeline that:

1. Loads evaluation examples from JSON or YAML.
2. Sends candidate responses to a judge model.
3. Evaluates responses against six quality criteria.
4. requests structured JSON judgments.
5. Validates and parses the judge's output.
6. Calculates weighted scores in Python.
7. Compares two candidate responses or model configurations.
8. Runs experiments to investigate possible judging biases.
9. Measures agreement with reference labels.
10. Saves evaluation results and logs for analysis.

The current implementation uses the local `llama3.2:3b` model through Ollama.

### Problem Statement

LLM-based evaluation can be inconsistent. A judge may prefer the first response it sees, reward unnecessarily long answers, produce malformed output, or disagree with reference judgments.

The goal of this project is to build a reproducible pipeline that measures these behaviors instead of assuming that an LLM judge is always reliable.

### Research Questions

- Can an LLM judge evaluate responses using a fixed, explicit rubric?
- Does changing the order of candidate responses change the selected winner?
- Does adding unnecessary text affect a response's evaluation score?
- Do different generation configurations produce different evaluation outcomes?
- How consistently does the judge evaluate the same examples?
- How closely do judge decisions agree with reference labels?

---

## Objectives

The primary objectives are:

- Build a runnable LLM-as-a-Judge evaluation pipeline.
- Define a structured, weighted evaluation rubric.
- Handle invalid or malformed judge responses.
- Measure position bias through candidate-order swaps.
- Measure sensitivity to verbosity through padded-response probes.
- Compare two generation configurations using a common evaluation process.
- Measure repeated-judgment consistency.
- Compare predictions with reference labels.
- Save reproducible artifacts and document the findings.

---

## Features

### Structured Evaluation

Each judgment contains:

- Case identifier
- Per-criterion score
- Rationale
- Supporting evidence, when available
- Overall weighted score
- Pairwise winner
- Decision rationale
- Confidence or diagnostic flags, when provided by the judge

### Robust Output Handling

The evaluator requests JSON output, validates it against a Pydantic schema, and retries when the output cannot be parsed or validated.

Retries improve robustness but do not guarantee that every judgment is correct.

### Configurable Models

The generator and judge have separate configuration fields. The current experiment uses the same local model for both roles, but the architecture allows the model settings to be changed independently.

### Bias Experiments

The pipeline includes:

- Candidate-order swapping for position bias
- Padded-response testing for verbosity sensitivity
- Repeated judgments for consistency
- An adversarial unsupported-claim probe

### Reproducible Artifacts

Evaluation outputs and diagnostic logs are written to local files so that results can be inspected after execution.

---

## Architecture

```text
                    +----------------------+
                    | JSON / YAML Test     |
                    | Suite (3 Cases)      |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    | Suite Loader         |
                    | Validate Case IDs    |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    | Evaluation Pipeline  |
                    +----------+-----------+
                               |
                 +-------------+-------------+
                 |                           |
                 v                           v
       +-------------------+       +-------------------+
       | Candidate A       |       | Candidate B       |
       | Model Output      |       | Model Output      |
       +---------+---------+       +---------+---------+
                 |                           |
                 +-------------+-------------+
                               |
                               v
                    +----------------------+
                    | LLM Judge            |
                    | Ollama                |
                    | llama3.2:3b           |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    | JSON Parser          |
                    | Pydantic Validation  |
                    | Retry on Invalid JSON|
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    | Weighted Scoring     |
                    | Python               |
                    +----------+-----------+
                               |
             +-----------------+------------------+
             |                 |                  |
             v                 v                  v
     +---------------+ +---------------+ +----------------+
     | Bias Analysis | | A/B Comparison| | Validation and |
     |               | |               | | Consistency    |
     +-------+-------+ +-------+-------+ +--------+-------+
             |                 |                  |
             +-----------------+------------------+
                               |
                               v
                    +----------------------+
                    | JSON Reports         |
                    | Logs and Metrics     |
                    +----------------------+
```

### Main Components

| Component | Responsibility |
|---|---|
| Suite loader | Reads and validates JSON/YAML evaluation cases |
| Configuration | Loads model settings, thresholds, and report paths |
| Provider | Communicates with the local Ollama API |
| Evaluator | Builds judge prompts, requests judgments, validates results |
| Parser and schemas | Validates structured judge responses |
| Rubric | Defines criteria and their weights |
| Metrics | Calculates aggregate scores, pass rates, and agreement |
| Bias module | Evaluates sensitivity to candidate order and verbosity |
| A/B comparison | Compares two generator configurations |
| Consistency module | Repeats evaluations to measure stability |
| Adversarial module | Tests sensitivity to an unsupported claim |
| Validation module | Compares judgments against reference labels |
| CLI | Provides commands to run experiments |

---

## Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.12 | Main implementation language |
| Ollama | Local model inference |
| `llama3.2:3b` | Current generator and judge model |
| Pydantic | Structured data validation |
| PyYAML | YAML configuration and dataset loading |
| HTTPX | HTTP requests to Ollama |
| pytest | Automated testing |
| JSON / JSONL | Test data, results, and logs |
| Git and GitHub | Version control and repository hosting |

The current pipeline is designed to run locally and does not require a paid model API.

---

## Repository Structure

```text
judgebench-llm-evaluation/
├── configs/
│   ├── config.yaml
│   ├── config_a.yaml
│   └── config_b.yaml
│
├── data/
│   ├── suite.json
│   ├── suite.yaml
│   └── gold_labels.json
│
├── logs/
│   └── judge_logs.jsonl
│
├── reports/
│   ├── evaluation_results.json
│   ├── position_results.json
│   ├── verbosity_results.json
│   ├── all_results.json
│   └── ab_comparison.json
│
├── src/
│   └── judgebench/
│       ├── __init__.py
│       ├── schemas.py
│       ├── config.py
│       ├── rubric.py
│       ├── parser.py
│       ├── providers.py
│       ├── evaluator.py
│       ├── bias.py
│       ├── metrics.py
│       ├── validation.py
│       ├── report.py
│       ├── cli.py
│       ├── ab_compare.py
│       ├── suite.py
│       ├── consistency.py
│       └── adversarial.py
│
├── tests/
│   ├── test_parser.py
│   ├── test_metrics.py
│   ├── test_bias.py
│   ├── test_validation.py
│   ├── test_suite.py
│   ├── test_consistency.py
│   └── test_adversarial.py
│
├── .github/
│   └── workflows/
│       └── tests.yml
│
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md
└── DISCUSSION.md
```

**Note:** The reports directory may contain additional files generated by experiments. The exact filenames depend on the commands and output paths used.

---

## Prerequisites

Install or prepare:

- Ubuntu or another supported operating system
- Python 3.12
- Git
- Ollama
- The local `llama3.2:3b` model

Check the installed versions:

```bash
python3 --version
git --version
ollama --version
```

---

## Installation

Clone the repository:

```bash
git clone git@github.com:student-kshitish/judgebench-llm-evaluation.git
cd judgebench-llm-evaluation
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Upgrade pip and install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If the virtual environment already exists, activate it instead of recreating it.

---

## Configure Ollama

Ensure Ollama is installed and running.

Check available local models:

```bash
ollama list
```

If `llama3.2:3b` is already installed, no additional model download is needed.

Otherwise, install it:

```bash
ollama pull llama3.2:3b
```

Test model inference:

```bash
ollama run llama3.2:3b
```

Enter a short question to verify that the model responds. Exit the interactive session when finished.

The default Ollama endpoint is:

```text
http://localhost:11434
```

The main configuration is stored in `configs/config.yaml`.

Example configuration:

```yaml
ollama:
  base_url: "http://localhost:11434"
  timeout_seconds: 180

generator:
  provider: "ollama"
  model: "llama3.2:3b"
  temperature: 0.2

judge:
  provider: "ollama"
  model: "llama3.2:3b"
  temperature: 0.0

evaluation:
  pass_threshold: 3.5
  score_scale_min: 1
  score_scale_max: 5
  repeat_count: 3

logging:
  directory: "logs"

reports:
  directory: "reports"
```

The actual configuration file is authoritative if it differs from this example.

The generator temperature controls the generation configuration. The judge temperature controls judge inference. Temperature zero is not a guarantee of perfectly deterministic results.

---

## Dataset Format

The pipeline supports JSON and YAML test suites.

The current dataset contains three cases:

- `capital_001`
- `math_001`
- `instruction_001`

The test suite is intentionally kept small for this experiment.

Each case contains an identifier and the information needed to evaluate candidate responses. Depending on the actual suite entry, fields can include an input, system prompt, candidate output, expected answer, and evaluation criteria.

Example of a conceptual JSON test case:

```json
{
  "id": "capital_001",
  "input": "What is the capital of France?",
  "system_prompt": "Answer accurately and concisely.",
  "model_output": "Paris is the capital of France.",
  "expected_output": "Paris",
  "criteria": [
    "correctness",
    "faithfulness",
    "completeness",
    "instruction_following",
    "tone_clarity",
    "safety"
  ]
}
```

This example illustrates the concept; use the field names and structure expected by the existing suite loader when modifying the real dataset.

### JSON Suite

```text
data/suite.json
```

### YAML Suite

```text
data/suite.yaml
```

### Reference Labels

```text
data/gold_labels.json
```

The reference labels provide comparison targets for the validation experiment. Unless independently reviewed by humans, they should be treated as provisional reference labels rather than authoritative human judgments.

---

## Evaluation Rubric

The judge scores responses on six criteria, using a scale from 1 to 5.

| Criterion | Weight | Description |
|---|---:|---|
| Correctness | 0.30 | Factual and logical accuracy |
| Faithfulness | 0.20 | Whether the answer is supported by the input and avoids unsupported claims |
| Completeness | 0.15 | Whether the answer addresses the important parts of the request |
| Instruction-following | 0.20 | Whether the answer follows the specified task and constraints |
| Tone and clarity | 0.05 | Whether the response is understandable and appropriately presented |
| Safety | 0.10 | Whether the response avoids unsafe or inappropriate assistance |
| **Total** | **1.00** | |

### Weighted Score

For criterion scores \(s_i\) and weights \(w_i\), the weighted score is:

\[
S = \sum_{i=1}^{6} w_i s_i
\]

where:

\[
\sum_{i=1}^{6} w_i = 1
\]

For example, if a response receives:

- Correctness: 5
- Faithfulness: 4
- Completeness: 4
- Instruction-following: 5
- Tone and clarity: 4
- Safety: 5

then:

\[
S = (0.30 \times 5)
+ (0.20 \times 4)
+ (0.15 \times 4)
+ (0.20 \times 5)
+ (0.05 \times 4)
+ (0.10 \times 5)
\]

\[
S = 4.60
\]

The weighted score is calculated in Python rather than relying on the judge to perform the final arithmetic.

The configured pass threshold is 3.5 on a 1–5 scale. A passing score is an operational threshold, not proof that the response is factually correct or safe in every context.

---

## Running the Pipeline

Activate the environment:

```bash
cd ~/judgebench-llm-evaluation
source .venv/bin/activate
```

Ensure Ollama is running and the configured model is available.

### 1. Run the Standard Evaluation

```bash
PYTHONPATH=src python -m judgebench.cli --mode evaluate
```

This runs the standard evaluation workflow.

The default output path is:

```text
reports/evaluation_results.json
```

### 2. Run Position Bias Evaluation

```bash
PYTHONPATH=src python -m judgebench.cli --mode position
```

This experiment changes the order in which candidate responses are presented and checks whether the selected winner changes.

Default output:

```text
reports/position_results.json
```

### 3. Run Verbosity Evaluation

```bash
PYTHONPATH=src python -m judgebench.cli --mode verbosity
```

This experiment compares original responses with padded versions to investigate whether unnecessary length changes the score.

Default output:

```text
reports/verbosity_results.json
```

### 4. Run the Combined Evaluation

```bash
PYTHONPATH=src python -m judgebench.cli --mode all
```

This runs the combined evaluation and diagnostic workflow.

Default output:

```text
reports/all_results.json
```

### Run with an Explicit Suite

For example:

```bash
PYTHONPATH=src python -m judgebench.cli \
  --suite data/suite.json \
  --mode all \
  --output reports/final_results.json
```

The precise command-line options supported by the installed code can be checked with:

```bash
PYTHONPATH=src python -m judgebench.cli --help
```

---

## Position Bias Experiment

### Objective

Position bias occurs when a judge's decision changes because the order of candidate responses changes, even though the substantive answers remain the same.

### Method

1. Present candidate A and candidate B to the judge.
2. Record the selected winner.
3. Swap the candidates' positions.
4. Run the judgment again.
5. Compare the decisions.

### Flip Rate

\[
\text{Flip Rate}
=
\frac{\text{Number of comparisons whose winner changes}}
{\text{Number of comparisons tested}}
\]

A lower flip rate indicates greater stability under the tested order swap. It does not establish that the judge is unbiased in other settings.

### Preliminary Result

An earlier run produced:

- Test cases: 3
- Position flip rate: 0.3333
- Approximate percentage: 33.33%

This means that one out of the three tested cases changed its selected winner in that run.

Because the dataset contains only three cases, the estimate is highly sensitive to individual outcomes.

To reproduce the experiment, run the position-bias command and inspect the generated report.

---

## Verbosity Bias Experiment

### Objective

This experiment investigates whether a judge rewards a longer response simply because it contains more text.

### Method

1. Evaluate the original candidate response.
2. Create a padded version with additional text.
3. Evaluate the padded response under the same general judging procedure.
4. Calculate the difference between the scores.

### Score Change

\[
\Delta S = S_{\text{padded}} - S_{\text{original}}
\]

Interpretation:

- \(\Delta S > 0\): the padded response received a higher score.
- \(\Delta S < 0\): the padded response received a lower score.
- \(\Delta S = 0\): the score did not change.

A score increase does not automatically prove verbosity bias. The added text must be checked to ensure that it does not improve the answer's actual completeness or usefulness.

### Preliminary Result

An earlier combined run reported:

- Test cases: 3
- Mean score change: approximately +0.0667

Another earlier run reported a mean change of approximately +0.10.

These are separate observed runs and should not be presented as one definitive result. Differences may reflect stochastic model behavior or changes in generated outputs.

Use the final report generated by the current experiment as the source of truth for the results submitted with the project.

---

## A/B Configuration Comparison

### Objective

Compare two generation configurations under the same evaluation suite.

The current configurations use the same model but different generator temperatures.

| Setting | Configuration A | Configuration B |
|---|---|---|
| Model | `llama3.2:3b` | `llama3.2:3b` |
| Generator temperature | 0.0 | 0.7 |
| Judge model | `llama3.2:3b` | `llama3.2:3b` |
| Dataset | Same three cases | Same three cases |

This comparison tests two generation settings. It is not a comparison between two different model families.

### Run the Comparison

```bash
PYTHONPATH=src python -m judgebench.ab_compare \
  --suite data/suite.json \
  --output reports/ab_comparison.json
```

Inspect the report:

```bash
python -m json.tool reports/ab_comparison.json
```

### Preliminary Result

An earlier run produced:

| Metric | Result |
|---|---:|
| Total cases | 3 |
| Successful cases | 3 |
| Failed cases | 0 |
| Configuration A mean score | 4.6833 |
| Configuration B mean score | 3.4833 |
| Observed winner | A |
| Mean-score difference | 1.2000 |

For that run, configuration A received the higher mean judge score.

This result is preliminary. With three cases and the same model family acting as generator and judge, it does not establish that temperature 0.0 is universally better than temperature 0.7.

The judge may also have preferences that affect both configurations. Independent human evaluation and additional test cases would be needed to support a stronger conclusion.

---

## Consistency Testing

### Objective

Repeated judgments help measure how stable the judge is when the same evaluation task is performed multiple times.

### Run

```bash
PYTHONPATH=src python -m judgebench.consistency \
  --suite data/suite.json \
  --repeats 3 \
  --output reports/final_consistency.json
```

The consistency module repeats judgments and produces a report containing the measurements implemented by the current code, such as winner agreement and score variability.

### Interpretation

- Higher winner agreement indicates that repeated judgments more often select the same winner.
- Lower score standard deviation indicates less variation in the numerical scores.
- Stable results do not guarantee correctness: a judge can consistently make the same mistake.

The report should be inspected before quoting exact consistency values in a submission.

---

## Adversarial Evaluation

### Objective

The adversarial probe tests whether adding an unsupported claim affects a candidate's score or pairwise result.

This helps investigate whether the judge penalizes unsupported information rather than simply preferring a more elaborate response.

### Run

```bash
PYTHONPATH=src python -m judgebench.adversarial \
  --suite data/suite.json \
  --output reports/final_adversarial.json
```

Inspect the resulting JSON report:

```bash
python -m json.tool reports/final_adversarial.json
```

### Interpretation

Examine whether the unsupported claim changes:

- The affected criterion scores
- The overall weighted score
- The selected winner
- The judge's rationale or diagnostic flags

A single adversarial probe is only a limited diagnostic. More carefully designed adversarial examples would be needed to characterize broader failure patterns.

---

## Validation Against Reference Labels

### Objective

The validation module compares the judge's decisions with reference labels stored in:

```text
data/gold_labels.json
```

Reference labels should ideally be produced or independently reviewed by human annotators.

### Metrics

**Exact agreement**

\[
\text{Agreement}
=
\frac{\text{Number of matching predictions}}
{\text{Number of evaluated labels}}
\]

**Cohen's kappa**

Cohen's kappa adjusts observed agreement for agreement expected by chance:

\[
\kappa = \frac{p_o - p_e}{1 - p_e}
\]

where:

- \(p_o\) is observed agreement.
- \(p_e\) is expected agreement under the metric's chance-agreement model.

Kappa can be unstable or difficult to interpret on very small datasets, especially when labels are imbalanced.

### Preliminary Result

An earlier run reported:

| Metric | Result |
|---|---:|
| Reference labels compared | 3 |
| Exact agreement | 0.6667 |
| Cohen's kappa | 0.0 |

These values describe that specific run and the current reference labels. The labels were not independently human-reviewed at the time of that report, so this is a comparison with provisional references, not verified human agreement.

A three-example result is too small to establish reliable judge validity.

---

## Metrics

The project includes or uses the following metrics, subject to the output of the current implementation.

| Metric | Purpose |
|---|---|
| Mean score | Average numerical evaluation score |
| Pass rate | Fraction of evaluated responses meeting the configured threshold |
| Winner counts | Number of decisions assigned to each candidate or a tie |
| Position flip rate | Fraction of tested comparisons whose winner changes after swapping candidate order |
| Verbosity score change | Difference between padded and original response scores |
| Exact agreement | Fraction of judge decisions matching reference labels |
| Cohen's kappa | Agreement adjusted for chance |
| Score standard deviation | Variability across repeated scores |
| Repeated-winner agreement | Stability of pairwise decisions across repeated evaluations |

Metrics should always be interpreted with the test-suite size, label quality, model configuration, and experimental procedure in mind.

---

## Logging and Reports

### Judge Logs

The evaluator logs judge prompts and raw responses in JSONL format. This makes it possible to inspect how a verdict was produced and diagnose malformed outputs.

The configured logging directory is:

```text
logs/
```

Logs may contain complete prompts and model-generated content. Review them before sharing the repository publicly if the data contains sensitive information.

### Reports

The configured report directory is:

```text
reports/
```

Examples of generated reports include:

```text
reports/
├── evaluation_results.json
├── position_results.json
├── verbosity_results.json
├── all_results.json
├── ab_comparison.json
├── final_results.json
├── final_ab_comparison.json
├── final_consistency.json
└── final_adversarial.json
```

The actual set of files depends on which experiments have been executed.

### Inspect a Report

```bash
python -m json.tool reports/all_results.json
```

For an A/B comparison:

```bash
python -m json.tool reports/ab_comparison.json
```

Use the actual report filenames generated by your commands. Do not report numbers for experiments that have not been run successfully.

---

## Preliminary Results

The following table consolidates results observed in earlier runs. These are historical preliminary observations, not a guarantee that a fresh run will reproduce identical values.

| Experiment | Preliminary observation | Interpretation |
|---|---|---|
| Combined pipeline | 9 successful evaluations and 0 failed operations | The recorded combined run completed its expected operations |
| Position bias | Flip rate of 0.3333 across 3 cases | One tested winner changed after the order swap |
| Verbosity sensitivity | Mean score change of approximately +0.0667 in one run | Padding was associated with a small positive mean score change in that run |
| A/B comparison | A: 4.6833; B: 3.4833 | Configuration A scored higher in that run |
| Reference-label agreement | 0.6667 across 3 labels | Two of the three decisions matched the reference labels in that run |
| Cohen's kappa | 0.0 | The observed agreement equaled the chance agreement estimated by kappa for that small sample |

### Key Observations

1. The pipeline completed its expected operations successfully in the recorded combined run.
2. At least one of the three cases was sensitive to candidate order.
3. The verbosity probe produced a small positive mean score change in one recorded run, but the result varied across runs.
4. Configuration A received a higher mean score than configuration B in the recorded A/B experiment.
5. The reference-label comparison indicates that agreement requires further investigation; the sample is too small to support a strong validity claim.

### Important Qualification

The dataset contains only three cases. Consequently:

- Results may change substantially when one case changes.
- The experiments cannot establish general performance across tasks or domains.
- Reference labels should be independently reviewed.
- The same-model generator/judge setup may introduce correlated preferences.
- Additional experiments are necessary before drawing broad conclusions.

The final submitted findings should be checked against the latest generated reports.

---

## Limitations

### Small Evaluation Dataset

Only three cases are used. This is sufficient for demonstrating the pipeline but not for reliable statistical generalization.

### Shared Generator and Judge Model

The current generator and judge use `llama3.2:3b`. This makes the system inexpensive to run locally, but the judge may favor response patterns associated with its own model family.

### Provisional Reference Labels

Reference labels have not been independently verified as human gold labels. Agreement metrics should therefore be interpreted cautiously.

### Stochastic Inference

Generation settings and inference behavior can affect outputs. Repeated runs may produce different scores and winners.

### Imperfect Rubric Scores

A judge can produce structured JSON and still make a factual or reasoning error. Schema validation guarantees structural constraints, not semantic correctness.

### Limited Bias Probes

Position swaps and padded responses test specific behaviors. They do not measure every possible form of bias, including familiarity bias, self-preference, calibration error, or sensitivity to answer style.

---

## Future Improvements

Potential improvements include:

1. Increase dataset size after the initial three-case prototype is complete.
2. Have independent human annotators review reference labels.
3. Use a different model family as the judge.
4. Add human-versus-judge agreement analysis on a larger labeled dataset.
5. Test more candidate-order permutations.
6. Add controlled verbosity probes that preserve semantic content.
7. Add explicit calibration and confidence analysis.
8. Improve judge prompt versioning and experiment metadata.
9. Record model versions, configuration hashes, and timestamps.
10. Add automated report generation and visualizations.
11. Test additional failure cases, including malformed JSON and unsupported claims.
12. Run experiments across multiple tasks and domains.

These are future directions rather than claims about features already completed.

---

## Running Tests

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Run the automated test suite:

```bash
PYTHONPATH=src pytest -v
```

Run a quiet compilation check:

```bash
python -m compileall -q src tests
```

A successful test run should show all tests passing. Record the actual test count and result from your current environment rather than assuming a fixed count.

---

## Troubleshooting

### Ollama Connection Error

Check whether Ollama is running:

```bash
ollama list
```

Check the local API:

```bash
curl http://localhost:11434/api/tags
```

If the API is unavailable, start or restart the Ollama service using the appropriate method for your installation.

### Model Not Found

Check:

```bash
ollama list
```

If required:

```bash
ollama pull llama3.2:3b
```

### Python Import Error

Make sure the virtual environment is active and run project commands with `PYTHONPATH=src`:

```bash
source .venv/bin/activate
PYTHONPATH=src python -m judgebench.cli --mode all
```

### Test Failure

Run:

```bash
PYTHONPATH=src pytest -v
```

Inspect the first failing test and its traceback before modifying code.

### Missing Report

Run the corresponding experiment again and check the output path printed by the command.

Do not assume a report exists just because its filename appears in this documentation.

---

## Conclusion

JudgeBench provides a local, configurable foundation for evaluating LLM responses with a structured rubric. It combines JSON/YAML test suites, schema-validated judgments, Python-based weighted scoring, A/B comparisons, bias probes, repeated evaluations, and reference-label comparisons.

The current implementation demonstrates how these components can be assembled into a reproducible evaluation workflow without requiring a paid inference API.

The observed results are preliminary because the current dataset contains only three cases and the reference labels have not been independently verified. The project should therefore be understood as a working evaluation prototype and an experimental investigation into judge reliability, rather than a definitive benchmark of LLM quality.

---

## License

Add a license file if you intend to distribute this project as open-source software. Until a license is selected and included in the repository, do not assume that the code is licensed for unrestricted reuse.
