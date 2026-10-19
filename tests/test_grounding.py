"""Quotation-grounding check (the computed anti-hallucination guard)."""

from cda.registers.grounding import check_quotations, render_grounding_markdown

PROMPT = (
    'Sample 1 [top1] [a.pdf]: "The standard story about scientific novelty '
    'treats it as a property of ideas that reviewers detect at submission." '
    'Sample 2 [top2] [b.pdf]: "Novelty is a core value in science, and a '
    'reliable measurement of novelty is crucial for evaluation research."'
)


def test_verified_quote_passes():
    body = (
        'The researcher writes: "The standard story about scientific novelty '
        'treats it as a property of ideas that reviewers detect at submission."'
    )
    result = check_quotations(body, PROMPT)
    assert result == {
        "n_quotes": 1,
        "n_verified": 1,
        "n_ambiguous": 0,
        "unverified": [],
        "unpaired_marks": 0,
    }


def test_planted_fabricated_quote_is_flagged():
    body = (
        'One real: "Novelty is a core value in science, and a reliable '
        'measurement of novelty is crucial for evaluation research." '
        'One planted: "Consensus formation requires seventeen distinct '
        'bureaucratic committees meeting weekly for years."'
    )
    result = check_quotations(body, PROMPT)
    assert result["n_quotes"] == 2
    assert result["n_verified"] == 1
    assert len(result["unverified"]) == 1
    assert "seventeen distinct" in result["unverified"][0]


def test_short_quotes_are_not_counted():
    body = 'The register "novelty as property" and the term "novelty" recur.'
    result = check_quotations(body, PROMPT)
    assert result["n_quotes"] == 0  # both under 6 words → not passage citations


def test_typography_and_whitespace_normalized():
    body = (
        "Quoted with curly quotes and extra spaces: “The standard story about\n"
        "   scientific novelty treats it as a property of ideas that reviewers "
        "detect at submission.”"
    )
    result = check_quotations(body, PROMPT)
    assert result["n_verified"] == 1


def test_ellipsis_spliced_quote_checks_each_fragment():
    good = (
        '"The standard story about scientific novelty treats it as a property '
        '… a reliable measurement of novelty is crucial for evaluation research."'
    )
    assert check_quotations(good, PROMPT)["n_verified"] == 0  # cross-source splice

    bad = (
        '"The standard story about scientific novelty treats it as a property '
        '… committees meeting weekly decide everything about novelty claims."'
    )
    result = check_quotations(bad, PROMPT)
    assert result["n_verified"] == 0 and result["n_quotes"] == 1


def test_render_all_verified():
    text = render_grounding_markdown({"n_quotes": 5, "n_verified": 5, "unverified": []})
    assert "**5/5**" in text
    assert "⚠️" not in text
    assert "verbatim" in text


def test_render_with_unverified_lists_previews():
    text = render_grounding_markdown(
        {"n_quotes": 3, "n_verified": 2, "unverified": ["a fabricated span here"]}
    )
    assert "**2/3**" in text
    assert "⚠️" in text
    assert "a fabricated span here" in text
    assert "does not mean fabricated" in text


def test_render_no_quotes():
    text = render_grounding_markdown({"n_quotes": 0, "n_verified": 0, "unverified": []})
    assert "No source-attributed or substantive quotations" in text


def test_quotes_never_span_paragraphs():
    """A short quoted term must not open a phantom match that swallows the
    text between two real quotations (the desynchronization bug caught on
    the real demo reports)."""
    body = (
        '# Report: "novelty"\n\n'
        "## Summary\n\n"
        'Two senses of "novelty" are found here in this analysis of usage.\n\n'
        '- "Novelty is a core value in science, and a reliable measurement '
        'of novelty is crucial for evaluation research."\n'
    )
    result = check_quotations(body, PROMPT)
    # Only the real ≥6-word citation is counted, and it verifies.
    assert result == {
        "n_quotes": 1,
        "n_verified": 1,
        "n_ambiguous": 0,
        "unverified": [],
        "unpaired_marks": 0,
    }
    # And nothing resembling markdown structure leaks into previews.
    assert all("##" not in preview for preview in result["unverified"])


def test_oversized_quote_is_reported_as_unchecked_not_all_clear():
    """A quotation beyond the pairing regex's span bound used to be neither
    verified nor listed — the report printed 'Every quotation … occurs
    verbatim' over a span the checker never looked at."""
    from cda.registers.grounding import check_quotations, render_grounding_markdown

    huge = "word " * 1200  # ~6000 chars, over the 4000-char span bound
    body = f'The model claims: "{huge.strip()}"\n'
    result = check_quotations(body, "prompt without it")
    assert result["unpaired_marks"] >= 2
    rendered = render_grounding_markdown(result)
    assert "NOT checked" in rendered
    assert "occurs verbatim" not in rendered


def test_balanced_reports_have_no_unpaired_marks():
    from cda.registers.grounding import check_quotations

    prompt = 'Sample 1 [top1] [a.pdf]: "novelty as a repair strategy in peer review debates"'
    body = 'The register quotes "novelty as a repair strategy in peer review debates" here.'
    result = check_quotations(body, prompt)
    assert result["unpaired_marks"] == 0
    assert result["n_verified"] == result["n_quotes"] == 1
