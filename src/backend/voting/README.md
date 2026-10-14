# Voting layer

LLM judges score generated acceptance criteria and UAT cases against four
rubrics: correctness, coverage, relevance and understandability.

To score a batch, call `run_voting_layer` (or `run_voting_layer_sync`) in
`voting_layer.py` with a list of inputs, each shaped like `EvaluationInput` in
`models.py`. Inputs are scored by the judges under `judges` in
config/models.json; pass `judge_ids` to use only some of them. The result is a
list of `VotingResult` objects, also saved as JSON.
