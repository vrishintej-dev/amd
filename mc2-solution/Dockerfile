# AMD AI League - Mini-Challenge 2 (OCR) submission
FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

# A 4B vision-language model: plenty for plates/signs, loads fast (the first image
# must not wait long for the model) and keeps the image ~38 GB.
ARG MODEL_ID=Qwen/Qwen3-VL-4B-Instruct
ENV MODEL_DIR=/models/vlm HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
WORKDIR /app

COPY app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
# torch-dependent: --no-deps so pip cannot replace ROCm torch with a CUDA build
RUN pip install --no-cache-dir --no-deps "transformers==5.21.0"
RUN python3 -c "import torch, transformers; v=torch.__version__; print('torch', v, 'transformers', transformers.__version__); assert '+rocm' in v, v"

COPY tools/download_model.py /tmp/download_model.py
RUN HF_HUB_OFFLINE=0 HF_HUB_DISABLE_XET=1 python3 /tmp/download_model.py "${MODEL_ID}" /models/vlm \
 && rm -rf /tmp/download_model.py /root/.cache/huggingface /models/vlm/.cache && du -sh /models/vlm

COPY app/ /app/
RUN mkdir -p /app/input /app/output

HEALTHCHECK --interval=5s --start-period=600s CMD test -S /tmp/mc2.sock
CMD ["sh", "-c", "python3 /app/server.py >> /tmp/mc2_server.log 2>&1; sleep infinity"]
