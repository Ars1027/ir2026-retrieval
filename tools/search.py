#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ir2026 import Corpus,CorpusTools

if __name__=='__main__':
    p=argparse.ArgumentParser(description='Literal search over the full raw corpus, with bounded JSON output.')
    p.add_argument('--terms',nargs='+',required=True);p.add_argument('--all',action='store_true');p.add_argument('--limit',type=int,default=10)
    a=p.parse_args()
    print(json.dumps(CorpusTools(Corpus(ROOT/'corpus'),'cli').search(a.terms,a.all,a.limit),ensure_ascii=False))
