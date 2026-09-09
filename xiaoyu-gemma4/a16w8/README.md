# Gemma 4 vision A16W8 calibration recipe

This recipe consumes `models/input/gemma4-vision.onnx` and runs:

1. `MatMulNBitsToQDQ`
2. `OnnxStaticQuantization` with UINT16 activations, UINT8 weights, QDQ format,
   and MinMax calibration
3. `DynamicToFixedShape` with `num_patches = 2520`

The calibration set contains 128 images sampled evenly and deterministically
from eight `HuggingFaceM4/the_cauldron` subsets:

- Natural photographs: `vqav2`, `visual7w`
- Diagrams and charts: `ai2d`, `chartqa`
- Documents and scene text: `docvqa`, `textvqa`
- Screenshots and science imagery: `screen2words`, `scienceqa`

Images are streamed from Hugging Face instead of downloading the complete
Cauldron dataset. Each image is resized with its aspect ratio preserved,
and converted to Gemma 4's 16x16 patches. Calibration samples retain their
natural, variable patch lengths up to the configured 2,520-patch budget
(`max_soft_tokens = 280`, with nine patches per soft token), so activation
quantization preserves the model's dynamic input. The final pass independently
fixes that input dimension to 2,520; removing that pass leaves the quantized
model dynamic.

## Linux GPU setup

From this recipe directory, link `models/input` to the directory containing
`gemma4-vision.onnx` and `model.onnx.data`:

```bash
mkdir -p models
ln -s /absolute/path/to/xiaoyu-gemma4 models/input
python -m pip install "olive-ai[gpu]>=0.13.0"
python -m pip install -r requirements.txt
```

## Run

Run from this directory so `user_script.py` and the model paths resolve
consistently:

```bash
olive run --config ./config.json
```

The quantized model is written under `models/a16w8`. Olive caches downloaded
data and intermediate pass output under `cache`.

To expand calibration after the 128-sample quality check, change
`samples_per_subset` in `config.json` from 16 to 32 or 64 for a total of 256
or 512 images.

The input model already contains 114 INT4 `MatMulNBits` nodes. Converting those
nodes to QDQ preserves their INT4 weights; the static pass applies W8
quantization to eligible floating-point weights and A16 quantization to
eligible activations. The resulting graph is therefore mixed W4/W8 rather
than uniformly W8.
