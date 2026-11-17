"""Run the pinned MLX CLI and emit measured memory from the same process."""
from __future__ import annotations

import json
import resource
import sys

import mlx.core as mx
from mlx_lm.lora import main


if __name__ == "__main__":
    main()  # Uses the original MLX argument parser and training implementation.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print("IR2026_RESOURCE_USAGE " + json.dumps({
        "peak_mlx_device_bytes": mx.get_peak_memory(),
        "peak_process_rss_bytes": rss if sys.platform == "darwin" else rss * 1024,
        "note": "MLX allocation peak and process RSS overlap on unified memory; do not add them.",
    }), flush=True)
