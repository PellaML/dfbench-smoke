FROM python:3.12-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    JAX_PLATFORMS=cpu \
    JAX_ENABLE_X64=true \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    XDG_CACHE_HOME=/tmp/cache \
    HOME=/tmp \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

RUN python -c "import ctypes; ctypes.CDLL('libstdc++.so.6')" \
    && dpkg-query -W libstdc++6

WORKDIR /app
COPY requirements-linux.lock ./
RUN python -m pip install --only-binary=:all: --no-deps --require-hashes -r requirements-linux.lock \
    && python -m pip check
COPY pyproject.toml README.md NOTICE ./
COPY third_party ./third_party
COPY src ./src
COPY tests ./tests
RUN python -m pip install --no-build-isolation --no-deps . \
    && python -m pip check
USER 65532:65532
ENTRYPOINT ["python", "-m", "dfbench_smoke"]
