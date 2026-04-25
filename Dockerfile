FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    torch \
    --extra-index-url https://download.pytorch.org/whl/cu124

RUN pip install --no-cache-dir unsloth

RUN pip install --no-cache-dir \
    transformers \
    accelerate \
    peft \
    trl \
    datasets \
    bitsandbytes

COPY infer.py .

ENV TORCH_COMPILE_DISABLE=1
ENV UNSLOTH_COMPILE_DISABLE=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV TORCH_CUDA_ARCH_LIST=9.0

CMD ["python", "infer.py"]
