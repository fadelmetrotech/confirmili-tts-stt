FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    git ffmpeg libsndfile1 build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python deps
RUN pip install --no-cache-dir runpod faster-whisper soundfile numpy

# F5-TTS (installs torchaudio from PyPI, which may be CPU-only)
RUN git clone https://github.com/SWivid/F5-TTS.git /app/F5-TTS && \
    cd /app/F5-TTS && pip install --no-cache-dir -e .

# Force-reinstall CUDA builds of torch+torchaudio AFTER F5-TTS
RUN pip install --no-cache-dir --force-reinstall \
    torch==2.2.0 torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu118

# Overwrite public F5-TTS model files with our custom emotion-aware versions
COPY dit.py /app/F5-TTS/src/f5_tts/model/backbones/dit.py
COPY cfm.py /app/F5-TTS/src/f5_tts/model/cfm.py

# Copy handler + darija utils
COPY handler.py /app/handler.py
COPY darija_utils.py /app/darija_utils.py
COPY dzd_to_darja.py /app/dzd_to_darja.py
COPY darija_french_dictionary.json /app/darija_french_dictionary.json

ENV MODELS_DIR=/runpod-volume/models

CMD ["python", "-u", "/app/handler.py"]
