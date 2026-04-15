# Annotation Guidelines

These rules apply when writing and evaluating question-answer pairs.

## Scope
- Questions must be factual and NBA-focused.
- Use publicly accessible sources only.
- Avoid opinion, prediction, or ambiguous phrasing.

## Answer Formatting
- One canonical answer per line in reference files.
- For multiple valid answers, separate alternatives with a semicolon.
- Keep spelling and naming consistent across files.

## Normalization Rules
- Treat case as non-essential during evaluation (case-insensitive matching).
- Ignore surrounding whitespace.
- Allow punctuation-insensitive comparison where possible.
- Use canonical names for people and teams.
- Use explicit units for quantities when needed (for example, minutes, feet).

## Ambiguity Handling
- If a question can have time-dependent answers, include a year/season in the question.
- If two answers are both valid, include both as semicolon-separated references.
- If annotators disagree, record rationale and adjudicate to one final reference line.

## Inter-Annotator Agreement
- Double-annotate a subset of items with at least two annotators.
- Compute Cohen's kappa on the shared subset.
- Document disagreements and final adjudication decisions.