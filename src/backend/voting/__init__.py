"""The voting layer: LLM judges score generated ACs and UAT cases.

Every judge rates each candidate against four rubrics: correctness,
coverage, relevance and understandability.

Import from the submodules directly. Nothing is re-exported here, so
that importing `voting.models` (as the app's config does) stays cheap
and doesn't pull in the MCP server.

Modules:

- `models`: judges, inputs, votes and results.
- `provider`: combined-style judging, one request for all rubrics.
- `prometheus`: per-rubric judging, one request per rubric.
- `voting`: runs every judge and merges their votes; also a script
  and MCP server.
- `voting_layer`: scores a batch and saves the results as JSON.
"""
