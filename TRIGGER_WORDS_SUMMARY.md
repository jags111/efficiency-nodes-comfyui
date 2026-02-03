# Summary: Trigger Words for LoRAs in XY Plot

## What's New?

You can now automatically add trigger words to your prompts when testing LoRAs in XY Plot workflows! This feature solves the problem where some LoRAs require specific trigger words to work effectively.

## Quick Start

### For Batch LoRA Testing

1. Use the **XY Input: LoRA Plot** node
2. In the `X_trigger_words` field, add one trigger word per line:
   ```
   anime style
   photorealistic
   oil painting
   ```
3. Your LoRAs will automatically get their trigger words during the batch run!

### For Individual LoRAs

1. Each LoRA slot in the **XY Input: LoRA** node now has a `trigger_word` field
2. Simply type the trigger word for each LoRA you select

### For LoRA Stacks

1. The **LoRA Stacker** node now has `trigger_word` fields for each slot
2. Create your stack with trigger words, and they'll be applied automatically

## Why Is This Useful?

**Before:** You had to either:
- Add all trigger words to your base prompt (causing unwanted interactions)
- Manually manage separate workflows for each LoRA
- Test without trigger words (suboptimal results)

**Now:** Trigger words are automatically added only when their specific LoRA is applied!

## Example

**Base Prompt:** "a beautiful landscape"

**LoRAs with Trigger Words:**
- style_lora_1.safetensors → "anime style, masterpiece"
- photo_lora.safetensors → "photorealistic, 8k"

**Automatic Results:**
- With style_lora_1: "a beautiful landscape anime style, masterpiece"
- With photo_lora: "a beautiful landscape photorealistic, 8k"

## Compatibility

✅ **Fully backward compatible** - existing workflows work without changes
✅ **Optional feature** - leave trigger words blank if you don't need them
✅ **Works with all XY Plot combinations** - LoRA weights, model strength, clip strength

## Where to Learn More

See [TRIGGER_WORDS_GUIDE.md](TRIGGER_WORDS_GUIDE.md) for detailed usage instructions, technical details, and troubleshooting tips.

## Feedback

If you encounter any issues or have suggestions for improvement, please open an issue on GitHub!
