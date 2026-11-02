"""Implement any retrieval method here, or export the same format independently.

solve(question, tools, config) receives no answer labels. The tools object exposes
search(terms, match_all=False, limit=10) and read(doc_id, start=0, max_chars=2000).
Return answer, ordered doc_ids, evidence, status and available model token usage.
For an external index, use tools.record_retrieval() to record actual stages.
The runner passes method_type and model in config. Models and harnesses are optional.
This student entry intentionally has no completed assignment implementation.
"""

def solve(question,tools,config):
    raise NotImplementedError('Implement your retrieval method, or export results according to SPEC.md.')
