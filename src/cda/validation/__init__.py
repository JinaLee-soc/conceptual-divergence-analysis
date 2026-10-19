"""Researcher-led validation of the clustering (EXPERIMENTAL, post-paper).

``cda validate`` opens a session in which the researcher records blind
verdicts on each cluster, compares them with independent LLM cluster
readings, and optionally curates the structure (merge/split) under agent
advice. The verdict trail is the validity evidence; curation never feeds
back into ``cda analyze``.
"""
