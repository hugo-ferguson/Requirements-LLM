# AI prompts

Prompt templates the voting judges read at run time. The models see this text
exactly as written, so change it only to change what the judges are told.

`voting_layer/` holds:

- `combined.txt`: the task and all four rubrics. `voting/provider.py` sends it
  as the system prompt for combined-style judges, so every vote shares it as a
  cacheable prefix.
- `combined_input.txt`: the message for each combined-style vote.
- `correctness.txt`, `coverage.txt`, `relevance.txt` and
  `understandability.txt`: one complete prompt per rubric, read by
  `voting/prometheus.py` for per-rubric (Prometheus-style) judges.

Before sending a template, the code replaces `{orig_instruction}` with the
prompt the candidates were generated from, and `{orig_response}` with the
candidate being scored.

The generation prompts aren't here. They're in `app/generation/prompts.py`.
