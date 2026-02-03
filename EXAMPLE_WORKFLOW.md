# Example Workflow: Using Trigger Words with LoRA Batch Testing

This document provides a step-by-step guide to create a workflow that tests multiple LoRAs with their respective trigger words.

## Scenario

You want to test 3 different style LoRAs on the same prompt to see which produces the best results. Each LoRA requires a specific trigger word.

## LoRAs to Test

1. `anime_style_v1.safetensors` → Trigger word: "anime style, masterpiece"
2. `photorealistic_v2.safetensors` → Trigger word: "photorealistic, 8k uhd"
3. `oil_painting.safetensors` → Trigger word: "oil painting, classical art"

## Step-by-Step Setup

### 1. Add Efficient Loader Node

**Settings:**
- `ckpt_name`: Your base model (e.g., "sd_v15.safetensors")
- `positive`: "a beautiful mountain landscape at sunset"
- `negative`: "low quality, blurry"
- Leave `lora_name` as "None" (we'll use the XY Plot instead)

### 2. Add XY Input: LoRA Plot Node

**Settings:**
- `input_mode`: "X: LoRA Batch, Y: LoRA Weight"
- `X_batch_path`: Path to your LoRA folder (e.g., "D:\LoRAs" or "/home/user/loras")
- `X_subdirectories`: false
- `X_batch_sort`: "ascending"
- `X_batch_count`: 3
- `model_strength`: 1.0
- `clip_strength`: 1.0

**NEW - Trigger Words Field:**
```
X_trigger_words:
anime style, masterpiece
photorealistic, 8k uhd
oil painting, classical art
```

**Important:** Make sure the trigger words are in the same order as your sorted LoRAs!

### 3. Add XY Input: LoRA Plot Node (for Y-axis)

For the Y-axis, we'll vary the LoRA weights:

**Settings:**
- This node provides the Y values for the weight variations
- Connect the Y output from the LoRA Plot node configured in step 2

OR use a separate simple value node if you prefer fixed weight steps.

### 4. Add XY Plot Node

**Settings:**
- Connect the `X` output from the LoRA Plot node to the XY Plot's `X` input
- Connect the `Y` output to the XY Plot's `Y` input
- `grid_spacing`: 5
- `XY_flip`: True (if you want LoRAs on X-axis)

### 5. Add KSampler (Efficient) Node

**Settings:**
- Connect `script` input to the XY Plot node's output
- Connect `model`, `positive`, `negative`, `latent` from the Efficient Loader
- Set your sampling parameters (steps, CFG, sampler, etc.)

### 6. Add Save Image Node

Connect the output from KSampler to save your results.

## Expected Results

When you run this workflow, you'll get an XY plot grid with:

**X-axis (LoRAs):**
- Column 1: Images generated with anime_style_v1 LoRA
  - Prompt used: "a beautiful mountain landscape at sunset anime style, masterpiece"
- Column 2: Images generated with photorealistic_v2 LoRA
  - Prompt used: "a beautiful mountain landscape at sunset photorealistic, 8k uhd"
- Column 3: Images generated with oil_painting LoRA
  - Prompt used: "a beautiful mountain landscape at sunset oil painting, classical art"

**Y-axis (Weights):**
- Varying LoRA strengths as configured

## Tips

1. **Verify LoRA Order:** Run the workflow with `X_batch_count: 1` first to verify which LoRA is loaded first, then adjust your trigger words accordingly.

2. **Empty Trigger Words:** If a LoRA doesn't need a trigger word, just leave that line blank:
   ```
   anime style, masterpiece
   
   oil painting
   ```
   (The second LoRA has no trigger word)

3. **Test Individually First:** Before running a large batch, test each LoRA individually with its trigger word to ensure you have the correct trigger words.

4. **Combine with Other XY Inputs:** You can also combine LoRA batching with checkpoint variations, sampler variations, etc.

## Troubleshooting

**Problem:** Trigger words aren't being applied
- **Solution:** Check that you've entered trigger words in the `X_trigger_words` field (multiline text area)

**Problem:** Wrong trigger word applied to wrong LoRA
- **Solution:** Verify your LoRAs are sorted in the expected order. Use the same sort order for trigger words.

**Problem:** Too many/too few trigger words
- **Solution:** The number of trigger words should match `X_batch_count`. Extra trigger words are ignored, missing ones default to empty.

## Advanced: Combining with Prompt S/R

You can use Prompt Search & Replace in combination with trigger words for even more control:

1. Set up your LoRA Plot with trigger words as above
2. Add an **XY Input: Prompt S/R** node for Y-axis instead
3. This allows you to vary both the LoRA (with its trigger word) and parts of the prompt simultaneously

Example:
- X-axis: Different LoRAs (each with trigger word)
- Y-axis: Replace "sunset" with ["sunrise", "midday", "midnight"]

Result: Each LoRA tested across different times of day, with appropriate trigger words applied.
