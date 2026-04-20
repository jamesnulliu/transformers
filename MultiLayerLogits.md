# 1. Multi-Layer Logits

This branch adds multi-layer logits support for the Qwen2 and Qwen3 causal language model forward paths.

## 1.1. Environment Parameters

The modifications in the last 4 commits use one environment variable:

### 1.1.1. `TARGET_LAYERS`

- Purpose: selects which decoder layers should be projected through `lm_head` and returned in `outputs.layer_logits`.
- Checked automatically in:
  - `src/transformers/__init__.py`
- Consumed in:
  - `src/transformers/models/qwen2/modular_qwen2.py`
  - `src/transformers/models/qwen2/modeling_qwen2.py`
  - `src/transformers/models/qwen3/modular_qwen3.py`
  - `src/transformers/models/qwen3/modeling_qwen3.py`
- Accepted formats:
  - `"0,1,2"`
  - `"[0,1,2]"`
  - whitespace is tolerated around values
- Required behavior:
  - if `TARGET_LAYERS` is unset, the code raises `ValueError`
  - if `TARGET_LAYERS` is set but empty, the code raises `ValueError`
  - the check happens automatically during `import transformers`
  - the error tells the user to set `TARGET_LAYERS` or avoid using this branch
- Validation:
  - if any index is `< 0` or `>= config.num_hidden_layers`, the code raises `ValueError`

Examples:

```bash
export TARGET_LAYERS=0,1,2
```

```bash
export TARGET_LAYERS='[4,8,16,24]'
```

## 1.2. Output Behavior

After these changes:

- `TARGET_LAYERS` is enforced at package import time before model code is used.
- `Qwen2Model` and `Qwen3Model` collect normalized hidden states from the selected layers only.
- `Qwen2ForCausalLM` and `Qwen3ForCausalLM` apply `lm_head` to each selected layer hidden state.
- The forward output becomes `CausalLMOutputWithPastAndLayerLogits`.
- Per-layer logits are returned in `output.layer_logits`, as a list ordered by `TARGET_LAYERS`.
- When `labels` are provided, loss is computed from `layer_logits[-1]`, meaning the last selected layer drives training loss.

## 1.3. Files Changed

### 1.3.1. `src/transformers/modeling_outputs.py`

- Added `BaseModelOutputWithPastAndLayerHiddenStates`
  - carries `layer_hidden_states`, `past_key_values`, `hidden_states`, and `attentions`
- Added `CausalLMOutputWithPastAndLayerLogits`
  - carries `loss`, `layer_logits`, `past_key_values`, `hidden_states`, and `attentions`
- Added docstrings describing the new output structures and shapes

### 1.3.2. `src/transformers/models/qwen2/modular_qwen2.py`

- No new branch-only assertion logic is added here.
- Changed `Qwen2Model.forward()` return type from `BaseModelOutputWithPast` to `BaseModelOutputWithPastAndLayerHiddenStates`
- Changed decoder loop to:
  - enumerate layers
  - collect `self.norm(hidden_states)` only for selected layer indices
  - return those collected hidden states instead of only final `last_hidden_state`
- Replaced the inherited `Qwen2ForCausalLM.forward()` behavior with custom logic that:
  - runs the model
  - computes logits for each collected layer hidden state
  - returns `CausalLMOutputWithPastAndLayerLogits`
  - computes loss from the last selected layer logits

### 1.3.3. `src/transformers/models/qwen2/modeling_qwen2.py`

- Generated sync of the `modular_qwen2.py` changes
- Contains the same functional changes as above in the generated modeling file
- Not a source-of-truth file; it is derived from the modular file

### 1.3.4. `src/transformers/models/qwen3/modular_qwen3.py`

- No new branch-only assertion logic is added here.
- Added a custom `Qwen3Model` implementation for this feature path
  - builds/uses cache and causal masks
  - iterates decoder layers
  - collects normalized hidden states for selected layers
  - returns `BaseModelOutputWithPastAndLayerHiddenStates`
- Replaced `Qwen3ForCausalLM.forward()` with custom logic that:
  - computes logits for each selected layer hidden state
  - returns `CausalLMOutputWithPastAndLayerLogits`
  - computes loss from the last selected layer logits
- Later fixes in the last 4 commits also synced Qwen3 behavior with Qwen2 and added target-layer index validation

### 1.3.5. `src/transformers/models/qwen3/modeling_qwen3.py`

- Generated sync of the `modular_qwen3.py` changes
- Contains the same functional changes as above in the generated modeling file
- Not a source-of-truth file; it is derived from the modular file

### 1.3.6. `src/transformers/__init__.py`

- Added an automatic import-time guard for `TARGET_LAYERS`
  - raises `ValueError` if `TARGET_LAYERS` is unset
  - raises `ValueError` if `TARGET_LAYERS` is empty/whitespace
  - tells the user to set `TARGET_LAYERS` or avoid using this branch
- This makes the branch fail fast without adding new manual validation calls in model definition files

## 1.4. Notes

- The last 4 commits only changed the 5 files listed above.
- There were no test or docs file changes in those commits.
- For future edits, `modular_qwen2.py` and `modular_qwen3.py` are the source-of-truth files; the generated `modeling_*.py` files should be refreshed from them.
