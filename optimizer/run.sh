#!/bin/bash

# CMD='RUSTFLAGS="-A warnings" cargo test --test keyformer count -- --nocapture'
CMD='RUSTFLAGS="-A warnings" cargo test --test basic -- --nocapture'

echo ">>> Running command: $CMD"
eval $CMD