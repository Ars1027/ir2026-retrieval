"""Build after member A provides chunks.jsonl and metadata.json."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrieval.dense import build_index

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/dense.json')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    settings = json.loads((root/args.config).read_text(encoding='utf-8'))
    source = root/settings['chunks_dir']
    chunks = [json.loads(line) for line in (source/'chunks.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    metadata = json.loads((source/'metadata.json').read_text(encoding='utf-8'))
    print(json.dumps(build_index(chunks, metadata, root/settings['dense_index_dir'], settings), indent=2))

if __name__ == '__main__':
    main()
