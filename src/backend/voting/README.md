# Voting layer

LLM judges score generated acceptance criteria and UAT cases against four
rubrics: correctness, coverage, relevance and understandability.

To score a batch, call `run_voting_layer` (or `run_voting_layer_sync`) in
`voting_layer.py` with a list of inputs, each shaped like `EvaluationInput` in
`models.py`. The enabled judges under `judges` in `config/models.json` score
them; pass `judge_ids` to use only some of them. It returns one `VotingResult`
per input, ranked by score and dumped to plain dicts, and also saves them as
JSON (by default to `votingLayerOutput.json` next to the module).

To serve the same scoring as the `evaluate_acceptance_criteria` MCP tool over
stdio, run `python -m voting.voting` from `src/backend`.
