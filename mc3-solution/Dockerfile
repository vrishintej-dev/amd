# AMD AI League - Mini-Challenge 3 (RAG) submission
# The FINAL stage must start from the mandated base (checked by layer identity).
# Never use --squash / FROM scratch tricks: they destroy the layer identity.
FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

ARG MODEL_ID=Qwen/Qwen3-VL-4B-Instruct
ENV MODEL_DIR=/models/vlm \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 1) Packages that do NOT depend on torch: normal install.
COPY app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 2) Packages that DO depend on torch: --no-deps, so pip cannot swap the ROCm torch
#    for a CUDA build (the trap described in the brief).
RUN pip install --no-cache-dir --no-deps "transformers==5.21.0"

# 3) Fail the build right here if torch is no longer the ROCm build.
RUN python3 -c "import torch, transformers; v=torch.__version__; print('torch', v, 'transformers', transformers.__version__); assert '+rocm' in v, v"

# 4) Ship the weights IN the image: there is no network at evaluation time.
#    (Network IS available here, at build time.) ~9 GB for the 4B VLM in bf16 (tested: 200/200 on the samples).
COPY tools/download_model.py /tmp/download_model.py
RUN HF_HUB_OFFLINE=0 HF_HUB_DISABLE_XET=1 python3 /tmp/download_model.py "${MODEL_ID}" /models/vlm \
 && rm -rf /tmp/download_model.py /root/.cache/huggingface /models/vlm/.cache && du -sh /models/vlm

COPY app/ /app/
RUN mkdir -p /app/corpus /app/output /app/index

# Start the resident worker (loads the model once). If it ever dies, keep the
# container alive anyway: the harness execs into it and app.py restarts it.
CMD ["sh", "-c", "python3 /app/server.py >> /tmp/mc3_server.log 2>&1; sleep infinity"]
