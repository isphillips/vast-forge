"""
Pre-download TRELLIS 2 weights INTO the image (HF_HOME=/models) so a cold Vast node boots warm instead of
pulling gigabytes on the first request. Fill MODEL_ID with the exact TRELLIS 2 repo id from its model card.
Runs at build time (see Dockerfile); failure is non-fatal so the image still builds while you wire the model.
"""
import os

MODEL_ID = os.environ.get("TRELLIS_MODEL_ID", "microsoft/TRELLIS.2-4B")

try:
    from huggingface_hub import snapshot_download

    path = snapshot_download(MODEL_ID)
    print(f"[vast-forge] pre-downloaded {MODEL_ID} -> {path}")
except Exception as e:  # noqa: BLE001 - best-effort at build time
    print(f"[vast-forge] weight pre-download skipped ({e}); set TRELLIS_MODEL_ID and ensure huggingface_hub is installed")
