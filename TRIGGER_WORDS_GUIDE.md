# Trigger Words for LoRAs in XY Plot

## Overview

This feature allows you to automatically add trigger words to your positive prompts when specific LoRAs are applied during XY Plot batch runs. This is particularly useful when testing multiple LoRAs that require specific trigger words to work effectively.

## How It Works

When a LoRA with a trigger word is applied during an XY Plot iteration, the trigger word is automatically appended to the positive prompt before the image is generated. This ensures that each LoRA gets its required trigger word without having to add all trigger words to the base prompt.

## Supported Nodes

The following nodes now support trigger words:

1. **XY Input: LoRA Plot** - For batch LoRA testing with varying weights
2. **XY Input: LoRA** - For individual LoRA selection
3. **LoRA Stacker** - For creating LoRA stacks with trigger words

## Usage

### XY Input: LoRA Plot (Batch Mode)

When using the LoRA Plot node in batch mode (e.g., "X: LoRA Batch, Y: LoRA Weight"):

1. **X_trigger_words** (multiline text field): Enter one trigger word per line, matching the order of your LoRAs in the batch directory.
   
   Example:
   ```
   anime style
   masterpiece, highly detailed
   photorealistic
   ```

2. The LoRAs will be sorted according to your `X_batch_sort` setting (ascending/descending), and trigger words will be matched to them in order.

3. If you have more LoRAs than trigger words, the extra LoRAs will have no trigger word (empty string).

### XY Input: LoRA Plot (Single LoRA Mode)

When testing a single LoRA with varying weights:

1. **trigger_word** (single line text field): Enter the trigger word for the selected LoRA.

   Example: `anime style, masterpiece`

2. This trigger word will be added to all iterations for that LoRA.

### XY Input: LoRA (Individual Selection)

When selecting individual LoRAs:

1. **trigger_word_1, trigger_word_2, etc.**: Each LoRA slot has its own trigger word field.

2. Enter the appropriate trigger word for each LoRA you select.

3. In batch mode, use the **trigger_words** (multiline) field instead.

### LoRA Stacker

When creating LoRA stacks:

1. **trigger_word_1, trigger_word_2, etc.**: Each LoRA in the stack has its own trigger word field.

2. These trigger words will be preserved when the stack is passed to other nodes.

## Example Workflow

Here's a typical workflow using trigger words:

1. Create an **XY Input: LoRA Plot** node
2. Set `input_mode` to "X: LoRA Batch, Y: LoRA Weight"
3. Set `X_batch_path` to your LoRA directory (e.g., `d:\LoRas`)
4. Set `X_batch_count` to the number of LoRAs you want to test
5. In the `X_trigger_words` field, enter trigger words (one per line):
   ```
   anime style, masterpiece
   photorealistic, 8k
   oil painting, classical art
   ```
6. Connect to your **XY Plot** node
7. Set up your base prompt in the **Efficient Loader** (e.g., "a beautiful landscape")
8. Run the workflow

**Result**: Each LoRA will be tested with its trigger word automatically added to the prompt:
- LoRA 1: "a beautiful landscape anime style, masterpiece"
- LoRA 2: "a beautiful landscape photorealistic, 8k"
- LoRA 3: "a beautiful landscape oil painting, classical art"

## Technical Details

### Data Structure

LoRA parameters are now stored as 4-tuples instead of 3-tuples:
- **Old format (backward compatible)**: `(lora_name, model_strength, clip_strength)`
- **New format**: `(lora_name, model_strength, clip_strength, trigger_word)`

The system automatically handles both formats for backward compatibility.

### Prompt Injection

Trigger words are appended to the positive prompt during XY Plot iteration, just before the model loads the LoRA and encodes the prompt. The original prompt is preserved in a tuple structure to support multiple iterations and combinations.

## Tips

1. **Empty trigger words are OK**: If a LoRA doesn't need a trigger word, just leave it blank.

2. **Multiple trigger words**: You can include multiple trigger words in a single field by separating them with commas: `anime style, masterpiece, highly detailed`

3. **Order matters**: In batch mode, make sure your trigger words are in the same order as your sorted LoRAs.

4. **Test first**: If you're unsure which trigger words a LoRA needs, check its documentation or test it individually first.

5. **Combining with Prompt S/R**: You can still use Prompt Search & Replace in combination with trigger words for even more control.

## Troubleshooting

**Q: My trigger words aren't being applied**
- Make sure you're using the updated nodes (check that trigger_word fields exist)
- Verify that you have trigger words entered in the correct fields
- Check that your LoRA count matches the number of trigger words (or use fewer trigger words)

**Q: Can I use trigger words with LoRA Stacks?**
- Yes! Use the LoRA Stacker node to create stacks with trigger words, then pass them to the XY Plot nodes.

**Q: Do trigger words work with the Efficient Loader's lora_name field?**
- The Efficient Loader's single LoRA field doesn't have a trigger word option. Use LoRA Stacker to create a stack with a trigger word, then connect it to the Efficient Loader's lora_stack input.

## Backward Compatibility

This feature is fully backward compatible. Existing workflows that don't use trigger words will continue to work exactly as before. The system automatically handles both 3-tuple (old) and 4-tuple (new) LoRA parameter formats.
