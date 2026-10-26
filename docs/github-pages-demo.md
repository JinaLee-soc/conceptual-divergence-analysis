# Fixed results web demo

Live demo: https://jinalee-soc.github.io/conceptual-divergence-analysis/

The GitHub Pages demo displays the committed v5 toy results for novelty and
consensus. Play advances through four explanatory stages; it is not an analysis
job or a historical timeline. The Streamlit demo remains available for running
the pipeline with recorded model responses.

## Build and preview

Python 3.11 or later is sufficient. The build uses the standard library only.
No CDA installation, corpus, model, provider key, or network call is required.
Choose a new output directory for each build; an existing directory is refused
to prevent unrelated files from entering the public artifact.

```sh
python scripts/build_pages.py --output /tmp/cda-public-preview
python -m http.server 8000 --directory /tmp/cda-public-preview
python -m unittest discover -s web/tests -v
```

The artifact contains HTML, CSS, JavaScript, two original Markdown reports, extracted display data, and file checksums. Paths are relative,
so the same files work at a GitHub Pages project URL or another static host.

## Publishing

Enable GitHub Actions as the Pages publishing source in repository Settings,
then set the Actions repository variable `PAGES_ENABLED` to `true`. Deployment
stays disabled until that explicit hosting configuration is complete.
The Public demo Pages workflow builds and tests on relevant changes; only main
can deploy. Pull requests build without deployment permissions. The workflow
uploads only the isolated build directory, never the repository root.

Keep repository visibility unchanged. Pages availability for private repositories
depends on the account plan. If GitHub rejects Pages creation, do not make the
repository public as a workaround; resolve the hosting choice with the maintainer.

## Source fidelity

The v5 source reports are listed explicitly in scripts/build_pages.py. The builder
extracts counts, cluster membership, quotations and checks directly from their
text, checks totals, and records their SHA-256 hashes. It does not rerun the
scientific analysis. HTML is escaped before bounded Markdown formatting.

The demonstration response was authored in a Claude Code session. Computed
sections describe the saved toy run. The UI keeps this provenance visible and
shows the low recurrence of the selected cluster counts. Quotation grounding
only checks matching against prompt passages, not scientific support.

## Privacy boundary

No uploads, input text, model calls, analytics, third-party scripts or external
fonts are used by the web page. GitHub may retain ordinary hosting access logs.
The public artifact excludes source code, Git history, private
corpus files and local output directories. Builds reject common private paths,
email addresses and credential patterns. Review new source reports before adding
them to the explicit publication allowlist; automated scanning is not a substitute
for reviewing the content. Use nonpersonal Git author and committer metadata for
site commits; do not rewrite pre-existing history.
