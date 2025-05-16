from transformers import AutoTokenizer, AutoModelForCausalLM
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

_MODEL = "models/Qwen3-14B-wenlv-ft"
_SAVE_DIR = f"{_MODEL}-FP8"

model = AutoModelForCausalLM.from_pretrained(_MODEL, device_map="auto", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(_MODEL)

# Configure the simple PTQ quantization
recipe = QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["re:.*lm_head"])

# Apply the quantization algorithm.
oneshot(model=model, recipe=recipe, output_dir=_SAVE_DIR)

