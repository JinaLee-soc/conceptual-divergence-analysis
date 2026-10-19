# CDA GUI image: slim Python, non-root, embedding model baked in.
# Build:  docker build -t cda .
# Run:    docker compose up   →  http://localhost:8501
# 3.11: the interpreter the dependency pins were resolved and tested on.
FROM python:3.11-slim

# Non-root user. Files created in the mounted volumes stay owned by the uid
# the container runs as — uid 1000 by default, or whatever a Linux host passes
# in through compose's `user:` (CDA_UID/CDA_GID) when its account is not 1000.
RUN useradd --create-home --uid 1000 cda
WORKDIR /app

# CPU-only torch first: the default PyPI wheel bundles CUDA (~multi-GB);
# the cpu wheel keeps the image small. torch is a transitive dependency
# (via sentence-transformers); 2.12.1 is what the pinned set resolves to
# in the tested build environment.
RUN pip install --no-cache-dir torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu

# Install CDA (pinned deps) + the GUI extra.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir ".[gui]"

# Everything the demo and GUI need at runtime.
COPY config ./config
COPY prompts ./prompts
COPY demo ./demo
COPY gui ./gui
COPY .streamlit ./.streamlit

# Bake the embedding model into the image so the demo needs no downloads.
ENV HF_HOME=/app/.hf-cache
# The chmod covers the two directories the app itself writes inside the image:
# the model cache (lock files) and the demo's local vector store. They are made
# writable by any uid so the run-as-your-own-uid escape hatch in
# docker-compose.yml (CDA_UID/CDA_GID, for Linux accounts that are not uid
# 1000) still gets a working demo button. The rest of /app stays read-only to
# that user.
# The revision is the same commit config/defaults.yaml pins (embedding.revision).
ARG EMBEDDING_REVISION=1110a243fdf4706b3f48f1d95db1a4f5529b4d41
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', revision='${EMBEDDING_REVISION}')" \
    && chown -R cda:cda /app \
    && chmod -R a+rwX /app/.hf-cache /app/demo
ENV HF_HUB_OFFLINE=1
# A foreign uid has no home directory in this image; /tmp is writable by
# everyone, so libraries that drop dotfiles in $HOME still work.
ENV HOME=/tmp

USER cda
EXPOSE 8501

# Streamlit GUI; volumes for ./corpus ./drafts ./outputs come from compose.
# 0.0.0.0 here is container-internal (the port mapping cannot reach a server
# bound to the container's loopback); what the outside world can reach is set
# by the `ports:` line in docker-compose.yml, which publishes on localhost.
CMD ["streamlit", "run", "gui/app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", \
     "--browser.gatherUsageStats=false"]
