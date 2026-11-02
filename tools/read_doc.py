#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ir2026 import Corpus,CorpusTools

if __name__=='__main__':
    p=argparse.ArgumentParser(description='Read a document range by original document ID.')
    p.add_argument('doc_id');p.add_argument('--start',type=int,default=0);p.add_argument('--max-chars',type=int,default=2000)
    a=p.parse_args()
    print(json.dumps(CorpusTools(Corpus(ROOT/'corpus'),'cli').read(a.doc_id,a.start,a.max_chars),ensure_ascii=False))
