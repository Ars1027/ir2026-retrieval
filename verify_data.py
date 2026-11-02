#!/usr/bin/env python3
import argparse,hashlib,json
from pathlib import Path
from ir2026 import Corpus,DataError,keyed,load_questions,read_jsonl

ROOT=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser(description='Check release hashes, split counts and labeled-document coverage.')
    p.add_argument('--full',action='store_true',help='Also hash every corpus document')
    args=p.parse_args()
    manifest=json.loads((ROOT/'DATA_MANIFEST.json').read_text())
    corpus=Corpus(ROOT/'corpus');seen=set()
    if len(corpus.docs)!=manifest['corpus_documents']:raise DataError('Corpus size mismatch')
    for rel,expected in manifest['data_file_sha256'].items():
        actual=hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()
        if actual!=expected:raise DataError(f'Changed data file: {rel}')
    for split,count in manifest['split_counts'].items():
        qs=load_questions(ROOT/'datas'/f'{split}.jsonl');ids=set(keyed(qs,'questions'))
        if len(ids)!=count or ids&seen:raise DataError('Invalid or overlapping split')
        seen|=ids
        if split!='test':
            labels=keyed(read_jsonl(ROOT/'datas'/f'{split}_labels.jsonl'),'labels')
            if set(labels)!=ids:raise DataError('Labels/query IDs mismatch')
            for label in labels.values():
                if not set(label['gold_doc_ids']+label['evidence_doc_ids'])<=set(corpus.docs):raise DataError('Missing positive document')
    if (ROOT/'datas/test_labels.jsonl').exists():raise DataError('Test labels must not be included in the student package')
    if args.full:
        bad=corpus.verify()
        if bad:raise DataError(f'{len(bad)} changed corpus documents')
    print(json.dumps({'valid':True,'release':manifest['release'],'questions':len(seen),'documents':len(corpus.docs),'full_hash_check':args.full},ensure_ascii=False))

if __name__=='__main__':main()
