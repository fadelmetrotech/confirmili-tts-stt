FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    git ffmpeg libsndfile1 build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python deps
RUN pip install --no-cache-dir runpod faster-whisper soundfile numpy
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# F5-TTS
RUN git clone https://github.com/SWivid/F5-TTS.git /app/F5-TTS && \
    cd /app/F5-TTS && pip install --no-cache-dir -e .

# Copy handler + darija utils
COPY handler.py /app/handler.py
COPY darija_utils.py /app/darija_utils.py
COPY dzd_to_darja.py /app/dzd_to_darja.py
COPY darija_french_dictionary.json /app/darija_french_dictionary.json

# Models go in /models (mounted via RunPod network storage)
ENV MODELS_DIR=/models

CMD ["python", "-u", "/app/handler.py"]
