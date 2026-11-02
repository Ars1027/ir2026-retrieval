"""Offline format demo, not a completed coursework solution or strong baseline."""
import re

STOP={'which','where','there','their','these','those','would','could','about','after','before',
      'during','what','when','with','from','that','this','were','have','been','name','named',
      'known','between','first','also','than','they','into','following','question','does'}

def solve(question,tools,config):
    terms=list(dict.fromkeys(w.lower() for w in re.findall(r'[A-Za-z][A-Za-z-]{3,}',question['question']) if w.lower() not in STOP))
    terms=sorted(terms,key=lambda t:(-len(t),t))[:6] or ['research']
    found=tools.search(terms,limit=10)['matches']
    ids=[r['doc_id'] for r in found]
    evidence=[]
    if found:
        r=found[0]
        read=tools.read(r['doc_id'],start=r['start'],max_chars=500)
        if len(read['text'])>=20:evidence=[{'doc_id':r['doc_id'],'text':read['text']}]
    return {'answer':'','doc_ids':ids if evidence else [],'evidence':evidence,'status':'no_answer',
            'usage':{'model_input_tokens':0,'model_output_tokens':0,'token_notes':'Offline keyword demo, no model calls.'}}
