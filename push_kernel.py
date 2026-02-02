import json
import os

metadata = {
  "id": "gabrielchavesreinann/nx47-arc-kernel",
  "title": "NX47 ARC KERNEL",
  "code_file": "aimo3_lum_enhanced_kernel.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": "true",
  "enable_gpu": "true",
  "enable_internet": "true",
  "dataset_sources": [],
  "competition_sources": ["ai-mathematical-olympiad-progress-prize"],
  "kernel_sources": []
}

with open("kernel-metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("Metadata created. Ready to push.")
