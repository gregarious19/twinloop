ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.09-py3
FROM ${BASE_IMAGE}
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt /tmp/requirements.txt
RUN python -c "import importlib.metadata as m; print('torch==' + m.version('torch'))" > /tmp/constraints.txt \
 && pip install -c /tmp/constraints.txt -r /tmp/requirements.txt \
 && python -c "import importlib.metadata as m; v=open('/tmp/constraints.txt').read().strip().split('==')[1]; assert m.version('torch') == v, (m.version('torch'), v)"
ENV PYTHONPATH=/workspace/twinloop/src:/workspace/twinloop
WORKDIR /workspace/twinloop
