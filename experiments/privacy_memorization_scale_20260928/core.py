import os,sys,json,time,hashlib,random
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
SNAP=ROOT/'.cache/huggingface/hub/models--zhihan1996--DNABERT-2-117M/snapshots/7bce263b15377fc15361f52cfab88f8b586abda0'
(OUT/'tmp').mkdir(exist_ok=True)
os.environ.update(HF_HOME=str(ROOT/'.cache/huggingface'),HF_MODULES_CACHE=str(OUT/'tmp/hf_modules'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',TEMP=str(OUT/'tmp'),TMP=str(OUT/'tmp'))
import numpy as np,pandas as pd,torch
from transformers import AutoModelForMaskedLM,AutoTokenizer
from transformers.models.bert.configuration_bert import BertConfig
torch.set_num_threads(4)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
CFG=read(OUT/'config.json')
def dump(path,x):
    path=OUT/path;path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def event(**x):print(json.dumps({'unix':time.time(),**x}),flush=True)
def seed_all(s):random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def tok():return AutoTokenizer.from_pretrained(SNAP,trust_remote_code=True,local_files_only=True)
def dna_ids(t):return sorted(v for k,v in t.get_vocab().items() if k and set(k)<=set('ACGT'))
def decode(t,ids):return ''.join(t.convert_ids_to_tokens([int(i) for i in ids]))
def rc(s):return s.translate(str.maketrans('ACGT','TGCA'))[::-1]
def canonical(s):return min(s,rc(s))
def fp32():torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
def load(seed=42,checkpoint=None):
    seed_all(seed);cfg=BertConfig.from_pretrained(SNAP,local_files_only=True);cfg.attention_probs_dropout_prob=.1
    m,info=AutoModelForMaskedLM.from_pretrained(SNAP,config=cfg,trust_remote_code=True,local_files_only=True,output_loading_info=True)
    assert not info['missing_keys'] and not info['mismatched_keys'],info
    if checkpoint:
        state=torch.load(checkpoint,map_location='cpu',weights_only=True);params=dict(m.named_parameters());assert set(params)==set(state)
        with torch.no_grad():
            for name,p in params.items():p.copy_(state[name])
    return m.cuda()
def pad(sequences,t,device='cuda'):
    ids=torch.full((len(sequences),64),t.pad_token_id,dtype=torch.long);att=torch.zeros_like(ids)
    for i,s in enumerate(sequences):
        row=[t.cls_token_id]+list(s)+[t.sep_token_id];assert len(row)<=64
        ids[i,:len(row)]=torch.tensor(row);att[i,:len(row)]=1
    return ids.to(device),att.to(device)
def mlm_batch(sequences,t,generator,allowed):
    ids,att=pad(sequences,t,'cpu');eligible=att.bool()&ids.ne(t.cls_token_id)&ids.ne(t.sep_token_id)
    selected=(torch.rand(ids.shape,generator=generator)<.15)&eligible
    if not selected.any():selected[0,1]=True
    labels=ids.clone();labels[~selected]=-100;draw=torch.rand(ids.shape,generator=generator)
    replacement=selected&(draw>=.8)&(draw<.9);ids[selected&(draw<.8)]=t.mask_token_id
    replacements=allowed[torch.randint(len(allowed),ids.shape,generator=generator)];ids[replacement]=replacements[replacement]
    return ids.cuda(),att.cuda(),labels.cuda(),int(selected.sum())
def model_dir(seed,arm):return OUT/'models'/f'seed_{seed}_{arm}'
def train_rows(data,arm):
    rows=[r['ids'] for r in data['background']]
    for row in data['contexts']:
        side='B' if arm=='B' or (arm=='single_flip' and row['id']==data['single_flip_id']) else 'A'
        rows.extend([row[f'ids_{side}']]*row['repeat'])
    assert len(rows)==3392
    return rows
def checkpoint_jobs():
    jobs=[]
    for seed in CFG['seeds']:
        for arm in CFG['arms']:
            for epoch in (5,10,20):jobs.append({'seed':seed,'arm':arm,'epoch':epoch,'name':f'{seed}_{arm}_e{epoch:02d}','path':model_dir(seed,arm)/f'epoch_{epoch:02d}.pt'})
        if seed in CFG['single_flip_seeds']:jobs.append({'seed':seed,'arm':'single_flip','epoch':20,'name':f'{seed}_single_flip_e20','path':model_dir(seed,'single_flip')/'epoch_20.pt'})
    return jobs
