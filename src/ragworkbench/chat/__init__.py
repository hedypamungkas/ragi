"""ragworkbench/chat -- ChatClient providers (single-turn text completion).

Used by Mode B eval (faithfulness judge + answer generation) and (later) query-rewrite.
Each provider satisfies the ``ChatClient`` Protocol (``async complete(messages) -> str``).
"""
