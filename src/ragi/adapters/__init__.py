"""ragi/adapters -- framework Retriever adapters.

Each adapter is behind its own extra (import-gated at module import) so the core stays
zero-dep. Import the adapter explicitly to trigger its gate::

    from ragi.adapters.langchain import LangChainRetrieverAdapter    # [adapters-langchain]
    from ragi.adapters.llamaindex import LlamaIndexRetrieverAdapter  # [adapters-llamaindex]
"""
