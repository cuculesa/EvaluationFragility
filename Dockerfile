ARG PYTHON_IMAGE=python:3.12-slim
ARG VCS_REF=unknown
ARG BUILD_DATE=unknown
FROM ${PYTHON_IMAGE} AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
COPY pyproject.toml README.md LICENSE /build/
COPY src /build/src
RUN python -m pip wheel --wheel-dir /wheels .

FROM ${PYTHON_IMAGE} AS runtime
ARG VCS_REF
ARG BUILD_DATE
LABEL org.opencontainers.image.title="EvalFrag" \
      org.opencontainers.image.description="Inspect AI evaluation-methodology sensitivity harness" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.created="${BUILD_DATE}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN groupadd --system evalfrag && useradd --system --gid evalfrag --create-home evalfrag
WORKDIR /app
COPY --from=builder /wheels /wheels
RUN python -m pip install --no-index --find-links=/wheels evalfrag && rm -rf /wheels
COPY configs /app/configs
RUN mkdir -p /app/data /app/runs && chown -R evalfrag:evalfrag /app
USER evalfrag
STOPSIGNAL SIGTERM
ENTRYPOINT ["evalfrag"]
CMD ["--help"]
