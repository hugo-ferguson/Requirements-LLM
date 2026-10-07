"""The voting layer: LLM judges score generated acceptance criteria and UAT
cases against four rubrics.

Import from the submodules directly. Nothing is re-exported here, so that
importing `voting.models` (as the app's config does) stays cheap and doesn't
pull in LiteLLM or the MCP server.
"""
