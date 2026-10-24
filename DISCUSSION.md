# Discussion: LLM-as-Judge Evaluation Pipeline

## 1. Overview

This project implements a local LLM-as-Judge evaluation pipeline for comparing candidate answers, measuring judge consistency, and investigating potential evaluation biases.

The system uses Ollama to run a local language model, `llama3.2:3b`, as the judge. The evaluation pipeline assesses candidate answers using a weighted rubric, produces structured verdicts, and records experiment results in JSON reports.

The project prioritizes reproducibility, structured outputs, automated testing, and local execution without requiring paid model APIs.

The current benchmark contains three test cases. Results are therefore preliminary and should not be interpreted as evidence of general judge reliability.

## 2. Research Objectives

The project investigates five main questions:

1. Can a local language model produce structured evaluation verdicts?
2. Does reversing candidate order change the judge's preferred answer?
3. Does adding unnecessary text influence candidate scores?
4. How consistently does the judge evaluate the same candidates across repeated runs?
5. Can unsupported claims influence the judge's score or winner selection?

An additional experiment compares two generation configurations to investigate whether generation temperature affects the quality of answers produced by the same model.

## 3. System Architecture

The implementation consists of the following components.

### 3.1 Test Suite

The test suite contains questions, candidate answers, expected answers or reference information, and evaluation criteria.

Both JSON and YAML test-suite formats are supported.

The current suite contains three cases covering factual correctness, arithmetic, and instruction following.

### 3.2 Candidate Generation

The A/B comparison module uses two independently configurable generation settings.

The current experiment uses `llama3.2:3b` with:

- Configuration A: temperature 0.0
- Configuration B: temperature 0.7

Both configurations use the same underlying model. This experiment compares generation settings rather than different model families.

### 3.3 LLM Judge

The judge evaluates both candidate answers using a structured rubric.

The rubric includes six criteria:

| Criterion | Weight |
|---|---:|
| Correctness | 0.30 |
| Faithfulness | 0.20 |
| Completeness | 0.15 |
| Instruction following | 0.20 |
| Tone and clarity | 0.05 |
| Safety | 0.10 |
| **Total** | **1.00** |

Each criterion receives a score from 1 to 5, together with a rationale and evidence.

The pipeline calculates the weighted candidate score programmatically instead of relying exclusively on the judge's self-reported overall score.

The overall score is calculated as:

\[
S = \sum_{i=1}^{6} w_i s_i
\]

where \(w_i\) is the weight of criterion \(i\), and \(s_i\) is its score.

The weights sum to 1.0, so the final weighted score remains on the 1–5 scale.

### 3.4 Structured Output and Error Handling

The pipeline extracts JSON from the judge's response and validates the result against Pydantic schemas.

Malformed or invalid outputs are retried. If evaluation continues to fail, the operation is recorded as a failure rather than being silently treated as a successful evaluation.

The system also records judge prompts, raw responses, model-call information, and experiment results.

### 3.5 Experiment Reports

Experiment outputs are saved as JSON files under the `reports/` directory.

This makes the results easier to inspect, compare, reproduce, and include in a final submission.

## 4. Experimental Methodology

### 4.1 Position-Bias Experiment

The position-bias experiment evaluates candidate answers in their original order and then reverses their positions.

The second verdict is mapped back to the original candidate identities before comparison.

The position-flip rate is:

\[
R_{\text{flip}} =
\frac{\text{Number of cases whose preferred candidate changes}}
{\text{Number of successfully tested cases}}
\]

A nonzero flip rate indicates that candidate order may influence decisions.

A zero flip rate would not prove that the judge is unbiased; it would only indicate that no flips were observed in the tested cases.

### 4.2 Verbosity-Bias Experiment

The verbosity experiment adds extra wording to a candidate answer and compares its score with the original score.

The score change is:

\[
\Delta S = S_{\text{modified}} - S_{\text{baseline}}
\]

A positive change indicates that the modified answer received a higher score. A negative change indicates a lower score.

This experiment tests one form of sensitivity to additional wording. It does not measure every possible form of verbosity bias.

### 4.3 A/B Configuration Comparison

The A/B experiment generates candidate answers using two configurations and evaluates them with a common judge.

The aggregate score for each configuration is the mean of its successful case-level scores.

The configuration with the higher aggregate score is declared the winner, unless the difference falls within the configured tie margin.

This result represents the judge's preference, not necessarily objective superiority. Independent reference labels and broader testing are needed to establish whether the preference reflects answer quality.

### 4.4 Repeated-Judgment Consistency

The consistency experiment evaluates the same candidates multiple times.

It records:

- The number of successful and failed judgments.
- Winner counts.
- Agreement with the first successful repetition.
- Agreement with the modal winner.
- Mean candidate scores.
- Score standard deviations.

Consistency measures repeatability, not correctness. A judge can repeatedly make the same incorrect decision.

### 4.5 Adversarial Robustness

The adversarial probe adds an unsupported claim that an answer has been independently verified by experts.

The system compares the original and modified evaluations to determine whether the claim changes the candidate's score or the selected winner.

The probe examines one specific failure mode. It does not establish general adversarial robustness.

## 5. Preliminary Experimental Results

The following observations come from the experiments performed during development.

### 5.1 Position Bias

The initial position-bias experiment reported:

- Cases tested: 3
- Position-flip rate: 0.3333, or approximately 33.3%

One of the three cases changed its preferred candidate after reversing the order.

This is a preliminary indication of possible position sensitivity. The small sample does not support a general conclusion about the judge's behavior.

### 5.2 Verbosity Bias

An earlier verbosity experiment reported the following score changes:

| Case | Score change |
|---|---:|
| Capital question | 0.00 |
| Arithmetic question | 0.00 |
| Instruction-following question | +0.30 |

The mean score change in that run was +0.10.

A subsequent run reported a mean change of approximately +0.067. These are results from separate runs, not a single combined experiment.

The results show that additional wording affected at least one case in the earlier run. They do not establish that longer answers are generally rewarded.

### 5.3 A/B Configuration Comparison

The reported A/B experiment produced:

| Metric | Configuration A | Configuration B |
|---|---:|---:|
| Model | `llama3.2:3b` | `llama3.2:3b` |
| Temperature | 0.0 | 0.7 |
| Mean judge score | 4.6833 | 3.4833 |

The observed difference was approximately 1.20 points in favor of Configuration A.

For this test suite, the judge preferred answers generated at temperature 0.0.

This finding is limited to the three tested cases, the specific model, and the chosen settings. It does not establish that temperature 0.0 is universally better.

### 5.4 Agreement with Reference Labels

The initial validation report recorded:

- Number of reference labels: 3
- Exact agreement: approximately 66.7%
- Cohen's kappa: 0.0

The agreement figure means that the judge matched two of the three reference decisions in that run.

The reference labels were provisional and require independent review. Consequently, these numbers should not be presented as validated human-judge agreement.

Cohen's kappa measures categorical agreement adjusted for agreement expected by chance. Its interpretation depends on the label distribution and sample size.

With only three cases, the estimate is highly unstable.

### 5.5 Repeated-Judgment and Adversarial Results

The repeated-judgment and adversarial experiments are implemented as separate modules.

Their final measured results should be taken directly from their generated JSON reports. They are not asserted here because no verified numerical summary for those runs is included in this discussion.

## 6. Limitations

### 6.1 Small Evaluation Dataset

The benchmark contains only three original cases.

Each case has a large influence on aggregate metrics. The dataset cannot represent the broad diversity of factual questions, reasoning tasks, writing tasks, safety scenarios, and instruction-following requirements encountered in real systems.

### 6.2 Provisional Reference Labels

Reference labels must be independently reviewed before being treated as human ground truth.

Using the same model's judgments as both predictions and reference labels would introduce circular evaluation.

### 6.3 Judge Reliability and Correctness Are Different

A judge may produce valid JSON, repeat the same decision, and still be wrong.

Structured output, operational success, consistency, and correctness are separate properties. They should be measured independently.

### 6.4 Model and Configuration Dependence

The experiments use a single local model family, `llama3.2:3b`.

The results may change with another model, a different prompt, a different temperature, or a different test-suite composition.

### 6.5 Limited Bias Coverage

The position, verbosity, and adversarial probes investigate specific forms of evaluation sensitivity.

They do not cover every possible issue, including self-enhancement bias, sycophancy, stylistic preferences, score clustering, and all forms of prompt sensitivity.

### 6.6 Subjective Scoring

Some rubric dimensions, such as completeness and tone, can require judgment.

Clearer scoring anchors, independently reviewed labels, and comparison against human evaluations would improve the validity of the results.

### 6.7 Local Model Constraints

Local execution avoids dependence on paid model APIs, but inference speed and output quality depend on available hardware and the model's capabilities.

The current implementation also requires additional work before it can be considered a production-grade evaluation service.

## 7. Future Work

The following improvements would strengthen the system:

1. Obtain independent human judgments and document the annotation procedure.
2. Add explicit score anchors for every rubric criterion.
3. Increase the diversity of test cases when a larger evaluation becomes feasible.
4. Measure repeated-run agreement across both winners and individual criterion scores.
5. Investigate score clustering and systematic preferences for particular writing styles.
6. Add self-enhancement and sycophancy probes.
7. Compare multiple judge models when additional local models are available.
8. Add category-level metrics and confidence intervals where sample sizes permit.
9. Improve experiment configuration tracking and result reproducibility.
10. Add continuous integration tests for all evaluation modules.

## 8. Conclusion

This project demonstrates a local, structured LLM-as-Judge evaluation workflow with JSON/YAML test suites, weighted rubric scoring, malformed-output handling, bias experiments, A/B comparison, agreement metrics, and repeated-judgment analysis.

The initial experiments demonstrate that the pipeline can produce measurable outputs and that its decisions can be investigated through controlled probes.

The observed position flips, verbosity score changes, and differences between generation configurations motivate further evaluation.

However, the three-case benchmark and provisional reference labels are insufficient to establish broad claims about judge quality or reliability.

The principal contribution at this stage is a reproducible experimental framework for investigating judge behavior, with explicit measurements and limitations rather than unsupported claims of accuracy.
