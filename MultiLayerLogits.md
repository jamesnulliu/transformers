# 1. Multi-Layer Logits

This branch adds multi-layer logits support for the Qwen2 and Qwen3 causal language model forward paths, with an optional CPU offload path for returned per-layer logits.

## 1.1. Environment Parameters

The current modifications use two environment variables:

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

### 1.1.2. `OFFLOAD_LAYER_LOGITS_TO_CPU`

- Purpose: controls whether returned `output.layer_logits` are detached and copied to CPU RAM before the forward output is returned.
- Consumed in:
  - `src/transformers/models/qwen2/modular_qwen2.py`
  - `src/transformers/models/qwen2/modeling_qwen2.py`
  - `src/transformers/models/qwen3/modular_qwen3.py`
  - `src/transformers/models/qwen3/modeling_qwen3.py`
- Accepted truthy values:
  - `"1"`
  - `"true"`
  - `"yes"`
  - `"on"`
- Accepted falsy values:
  - `"0"`
  - `"false"`
  - `"no"`
  - `"off"`
- Required behavior:
  - if unset or empty, the feature is disabled
  - if enabled, the code computes loss first and then returns `layer_logits` as detached CPU tensors
- if set to any other value, the code raises `ValueError`

### 1.1.3. `USE_HIDDEN_STATES`

- Purpose: controls whether returned `output.layer_logits` contain hidden states before `lm_head` instead of projected vocabulary logits.
- Consumed in:
  - `src/transformers/models/qwen2/modular_qwen2.py`
  - `src/transformers/models/qwen2/modeling_qwen2.py`
  - `src/transformers/models/qwen3/modular_qwen3.py`
  - `src/transformers/models/qwen3/modeling_qwen3.py`
- Accepted truthy values:
  - `"1"`
  - `"true"`
  - `"yes"`
  - `"on"`
- Accepted falsy values:
  - `"0"`
  - `"false"`
  - `"no"`
  - `"off"`
- Required behavior:
  - if unset or empty, the feature is disabled and `output.layer_logits` keeps returning post-`lm_head` vocabulary logits
  - if enabled, `output.layer_logits` returns the selected hidden states after layer norm and slicing, before `lm_head`
  - `output.logits` still returns final-layer post-`lm_head` vocabulary logits so inference and generation continue to work normally
  - when enabled and `labels` are provided, loss is computed from `output.logits`, so training loss remains vocabulary-based
  - if set to any other value, the code raises `ValueError`

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
- `Qwen2ForCausalLM` and `Qwen3ForCausalLM` apply `lm_head` to each selected layer hidden state by default.
- The forward output becomes `CausalLMOutputWithPastAndLayerLogits`.
- Per-layer outputs are returned in `output.layer_logits`, as a list ordered by `TARGET_LAYERS`.
- Final-layer vocabulary logits are returned in `output.logits`.
- By default, `output.layer_logits` contains projected vocabulary logits.
- If `USE_HIDDEN_STATES` is enabled, `output.layer_logits` contains the selected hidden states before `lm_head`.
- When `labels` are provided, loss is computed from `output.logits`, meaning the last selected layer still drives training loss even when hidden states are returned.
- If `OFFLOAD_LAYER_LOGITS_TO_CPU` is enabled, returned `output.layer_logits` are detached and copied to CPU after loss computation, regardless of whether they contain logits or hidden states.

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
  - computes logits for each collected layer hidden state by default
  - returns `CausalLMOutputWithPastAndLayerLogits`
  - computes loss from the last selected layer logits
- Added optional `OFFLOAD_LAYER_LOGITS_TO_CPU` parsing
  - when enabled, `layer_logits` are detached and copied to CPU before return
- Added optional `USE_HIDDEN_STATES` parsing
  - when enabled, `layer_logits` returns sliced hidden states before `lm_head`
  - if `labels` are provided, loss still uses `lm_head` on the last selected hidden state

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
  - computes logits for each selected layer hidden state by default
  - returns `CausalLMOutputWithPastAndLayerLogits`
  - computes loss from the last selected layer logits
- Later fixes in the last 4 commits also synced Qwen3 behavior with Qwen2 and added target-layer index validation
- Added optional `OFFLOAD_LAYER_LOGITS_TO_CPU` handling via the shared Qwen2 helper
  - when enabled, `layer_logits` are detached and copied to CPU before return
- Added optional `USE_HIDDEN_STATES` handling via the shared Qwen2 helper
  - when enabled, `layer_logits` returns sliced hidden states before `lm_head`
  - if `labels` are provided, loss still uses `lm_head` on the last selected hidden state

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

### 1.3.7. `tests/models/qwen2/test_modeling_qwen2.py`

- Added a focused unit test for `OFFLOAD_LAYER_LOGITS_TO_CPU`
  - verifies `layer_logits` are returned on CPU
  - verifies they are detached
  - verifies training loss is still produced
- Added a focused unit test for `USE_HIDDEN_STATES`
  - verifies `layer_logits` returns hidden-size tensors instead of vocab-size logits
  - verifies the returned tensors match `model.layer_hidden_states`
  - verifies training loss is still produced

### 1.3.8. `tests/models/qwen3/test_modeling_qwen3.py`

- Added a focused unit test for `OFFLOAD_LAYER_LOGITS_TO_CPU`
  - verifies `layer_logits` are returned on CPU
  - verifies they are detached
  - verifies training loss is still produced
- Added a focused unit test for `USE_HIDDEN_STATES`
  - verifies `layer_logits` returns hidden-size tensors instead of vocab-size logits
  - verifies the returned tensors match `model.layer_hidden_states`
  - verifies training loss is still produced

## 1.4. Notes

- `modular_qwen2.py` and `modular_qwen3.py` remain the source-of-truth files.
- The generated `modeling_qwen2.py` and `modeling_qwen3.py` files should be refreshed from the modular sources after edits.
- The CPU offload path is intended for logging/inspection of returned `layer_logits`, not as the default training path.
- `layer_logits` is now a branch-specific container name and may hold either projected logits or pre-`lm_head` hidden states depending on `USE_HIDDEN_STATES`.
