# RunPod Serverless Deployment

## 1. Upload models to RunPod Network Storage

Create a Network Volume on RunPod and upload:

```
/models/
  whisper-darija-ct2/
    config.json        ← from C:\Users\fadel\Desktop\audios\asr_finetune\whisper-darija-ct2\
    model.bin
    vocabulary.json
  tts/
    model_last.pt      ← from C:\confirmili_tts\F5-TTS\ckpts\confirmili_emotion\
    vocab.txt          ← from C:\confirmili_tts\ckpts\
    refs/
      darija_001083.wav
      darija_004964.wav
      darija_000794.wav
      darija_001602.wav
      darija_005314.wav
```

## 2. Build & push Docker image

```bash
cd C:\Users\fadel\Desktop\audios\runpod

# Copy required files
copy ..\darija_utils.py .
copy ..\dzd_to_darja.py .
copy ..\darija_french_dictionary.json .

# Build
docker build -t confirmili-tts-stt .

# Push to Docker Hub (or RunPod registry)
docker tag confirmili-tts-stt YOUR_DOCKERHUB/confirmili-tts-stt:latest
docker push YOUR_DOCKERHUB/confirmili-tts-stt:latest
```

## 3. Create Serverless Endpoint on RunPod

- Go to RunPod → Serverless → New Endpoint
- Docker image: `YOUR_DOCKERHUB/confirmili-tts-stt:latest`
- GPU: H100 SXM
- Attach your Network Volume at `/models`
- Min workers: 0 (pay per use)
- Max workers: 3

## 4. Configure call_simulator

Set environment variables before running:

```bash
set RUNPOD_ENDPOINT_ID=your_endpoint_id_here
set RUNPOD_API_KEY=your_runpod_api_key_here
python call_simulator.py
```

Or hardcode them in call_simulator.py lines:
```python
RUNPOD_ENDPOINT_ID = "your_endpoint_id"
RUNPOD_API_KEY     = "your_api_key"
```

When set → uses RunPod H100 (fast, ~0.3s TTS)
When None → uses local GPU (your RTX 4060)
