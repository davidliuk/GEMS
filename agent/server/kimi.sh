MODEL_PATH=

sglang serve --model-path $MODEL_PATH \
  --host 0.0.0.0 \
  --tp 8 \
  --trust-remote-code \
  --tool-call-parser kimi_k2 \
  --reasoning-parser kimi_k2 \
  --cuda-graph-max-bs 128 \
  --mem-fraction-static 0.9