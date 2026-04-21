# DreamZero Model Weights Location

## Weights Storage Path

All pretrained model weights are stored under:

```
/fact_data/qiliu/dreamzero_weights/
```

### DreamZero-DROID (Inference)

- **Path**: `/fact_data/qiliu/dreamzero_weights/DreamZero-DROID`
- **Source**: `GEAR-Dreams/DreamZero-DROID` (HuggingFace)
- **Usage**: Zero-shot inference on robot manipulation tasks
- **Model Size**: ~14B parameters, 43GB on disk (10 safetensors shards)

## How to Run Inference

Activate the virtual environment:

```bash
source ~/dreamzero_env/bin/activate
cd /fact_home/qiliu/worldmodel/dreamzero
```

Start the distributed inference server (using 2 GPUs):

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
  --standalone --nproc_per_node=2 \
  socket_test_optimized_AR.py \
  --port 5000 \
  --enable-dit-cache \
  --model-path /fact_data/qiliu/dreamzero_weights/DreamZero-DROID
```

Test with the client:

```bash
python test_client_AR.py --port 5000
```

## Notes

- The machine has 8x NVIDIA H20 (96GB each). Inference requires at least 2 GPUs.
- `--enable-dit-cache` is recommended for faster inference (~3s/frame on H20).
- Adjust `--nproc_per_node` and `CUDA_VISIBLE_DEVICES` to use more GPUs if needed.
