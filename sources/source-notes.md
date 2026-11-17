# Source Notes

Accessed dates below are 2026-11-17 unless stated otherwise.

## Primary research direction

### Anthropic — Recommendations for Technical AI Safety Research Directions

- URL: https://alignment.anthropic.com/2025/recommended-directions/
- Relevant idea: improve oversight despite systematic, exploitable errors in the oversight signal by distinguishing a true signal R\*, a flawed signal R, and evaluation against R\*.
- Relevance: motivates the study's in-context measurement of model behavior under systematically faulty evaluator feedback. The experiment does not train model weights.

## Related methods

### FindTheFlaws: Annotated Errors for Detecting Flawed Reasoning and Scalable Oversight Research

- URL: https://github.com/modulo-research/findtheflaws
- Relevance: distinguishes confirmation of correct reasoning from detection and localization of flawed reasoning.

### Evaluating Large Language Models at Evaluating Instruction Following (LLMBar)

- URL: https://github.com/princeton-nlp/LLMBar
- Relevance: meta-evaluates whether LLM evaluators can select instruction-following outputs under natural and adversarial comparisons.

### CriticEval: Evaluating Large Language Model as Critic

- URL: https://github.com/open-compass/CriticEval
- Relevance: separates feedback, comparison, refinement, and meta-feedback capabilities instead of treating critique as one undifferentiated score.
