"""Public order-2 DNA baseline; target secrets are not read."""
import sys,collections
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import core as c
import numpy as np,pandas as pd
t=c.tok();counts=collections.defaultdict(lambda:np.ones(4,dtype=np.float64));alphabet='ACGT'
frame=pd.read_csv(c.ROOT/'GUE_v2/EMP/H3K4me3/dev.csv')
# Fixed public sample, independent of secret construction and extraction results.
for index in np.random.default_rng(910999).permutation(len(frame))[:512]:
    s=str(frame.iloc[int(index)].sequence);s=s[(len(s)-96)//2:(len(s)-96)//2+96]
    for j,ch in enumerate(s):
        for k in range(min(2,j)+1):counts[s[j-k:j]][alphabet.index(ch)]+=1
def generate(prefix):
    generated='';context=prefix
    for _ in range(12):
        key=context[-2:] if len(context)>=2 else context
        while key not in counts and key:key=key[1:]
        ch=alphabet[int(np.argmax(counts[key]))];generated+=ch;context+=ch
    return generated
for seed in c.CFG['seeds']:
    rows=c.read(c.OUT/f'public/challenges_{seed}.json');pred={r['id']:generate(c.decode(t,r['prefix_ids'])) for r in rows if r['hidden_token_count']==3}
    c.dump(f'baselines/markov_{seed}.json',pred)
c.dump('markov_baseline_DONE.json',{'public_rows':512,'order':2,'pseudocount':1,'target_bp_length_provided':12,'target_secrets_read':False})
