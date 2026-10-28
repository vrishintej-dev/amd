# AMD AI League - Match 4 (Intelligent Web Information Retrieval)
# Built on our Mini Challenge 3 image, whose lower layers ARE the mandated
# rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0 base (identical
# layers), and which already ships Qwen3-VL-4B in /models/vlm. Only the code changes,
# so the push is a few MB instead of ~38 GB.
FROM docker.io/vrishintej/rag-e0beb53a:v1
ENV MODEL_DIR=/models/vlm HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
COPY app/ /app/
RUN mkdir -p /app/input /app/output /app/corpus /app/index \
 && python3 -c "import torch, requests; v=torch.__version__; print('torch', v); assert '+rocm' in v, v"
CMD ["sh", "-c", "python3 /app/server.py >> /tmp/mc4_server.log 2>&1; sleep infinity"]
