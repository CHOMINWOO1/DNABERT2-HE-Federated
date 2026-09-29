"""Blind conditional generation: no private data, labels, or membership read."""
import sys,time,gc,itertools,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import core as c
import numpy as np,torch

def deny_truth(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes)):
        p=str(args[0]).replace('\\','/').lower()
        if '/private/' in p or '/gue_v2/' in p or p.endswith('/metrics_records.json'):
            raise PermissionError('Blind attack cannot read raw data or evaluator truth: '+p)
sys.addaudithook(deny_truth)

def top_beams(values,width):
    n,b,v=values.shape;flat=values.reshape(n,-1);idx=torch.argsort(flat,dim=-1,descending=True,stable=True)[:,:width]
    return idx//v,idx%v,flat.gather(1,idx)

def check_algorithm():
    def lp(path):
        if not path:return np.log([.55,.45])
        if len(path)==1:return np.log([.5,.5] if path[0]==0 else [.01,.99])
        return np.log([.5,.5] if path[0]==0 else [.99,.01])
    paths=[()];score=torch.zeros(1,1)
    for _ in range(3):
        p,i,score=top_beams(torch.tensor(np.array([lp(x) for x in paths]))[None]+score[:,:,None],2)
        paths=[paths[int(a)]+(int(b),) for a,b in zip(p[0],i[0])]
    exhaustive=max(itertools.product(range(2),repeat=3),key=lambda p:sum(lp(p[:j])[p[j]] for j in range(3)))
    greedy=()
    for _ in range(3):greedy+=(int(lp(greedy).argmax()),)
    assert paths[0]==exhaustive and paths[0]!=greedy
    return {'toy_beam_matches_exhaustive':True,'toy_beam_beats_greedy':True,'truth_read_guard':True}

@torch.inference_mode()
def extract(m,challenges,t,width=1):
    assert all(set(r)=={'id','prefix_ids','hidden_token_count','context_fraction'} for r in challenges)
    m.float().eval();allowed=torch.tensor(c.dna_ids(t),device='cuda');unique={};mapping={}
    for row in challenges:
        key=(tuple(row['prefix_ids']),row['hidden_token_count']);unique.setdefault(key,row);mapping[row['id']]=key
    cache={};sequence_forwards=0
    for k in sorted(set(r['hidden_token_count'] for r in challenges)):
        rows=[r for r in unique.values() if r['hidden_token_count']==k]
        batch_size=32 if width==1 else 4
        for start in range(0,len(rows),batch_size):
            batch=rows[start:start+batch_size];n=len(batch);ids,att=c.pad([r['prefix_ids']+[t.mask_token_id]*k for r in batch],t)
            pos=torch.tensor([len(r['prefix_ids'])+1 for r in batch],device='cuda');seq=ids[:,None,:];mask=att[:,None,:]
            score=torch.zeros(n,1,device='cuda');beams=1;ties=[[] for _ in batch]
            for step in range(k):
                hidden=m.bert(input_ids=seq.reshape(n*beams,64),attention_mask=mask.reshape(n*beams,64))[0]
                logits=m.cls(hidden[torch.arange(n*beams,device='cuda'),(pos+step).repeat_interleave(beams)])
                assert logits.dtype==torch.float32 and torch.isfinite(logits).all()
                lp=torch.log_softmax(logits[:,allowed],-1).reshape(n,beams,-1);values=score[:,:,None]+lp
                top=values.flatten(1).max(1).values
                for j,value in enumerate((values.flatten(1)==top[:,None]).sum(1).cpu().tolist()):ties[j].append(value)
                parents,tokens,score=top_beams(values,width);idx=torch.arange(n,device='cuda')[:,None]
                seq=seq[idx,parents].clone();mask=mask[idx,parents].clone()
                seq[idx,torch.arange(width,device='cuda')[None,:],(pos+step)[:,None]]=allowed[tokens]
                sequence_forwards+=n*beams;beams=width
            for i,r in enumerate(batch):
                output=seq[i,0,pos[i]:pos[i]+k].cpu().tolist()
                cache[(tuple(r['prefix_ids']),k)]={'ids':output,'dna':c.decode(t,output),'selected_log_score':float(score[i,0]),'top_ties':ties[i]}
    return {cid:cache[key] for cid,key in mapping.items()},{'unique_public_inputs':len(unique),'challenge_count':len(challenges),'sequence_forwards':sequence_forwards}

def run_job(job,m,t,output_dir='predictions',replay=False):
    out=c.OUT/output_dir/f'{job["name"]}.json';done=out.with_suffix('.DONE.json')
    if done.exists() and not replay:
        assert c.sha(out)==c.read(done)['prediction_sha256'];return
    challenges=c.read(c.OUT/f'public/challenges_{job["seed"]}.json')
    broad=job['epoch']==20 and job['arm'] in ('A','B') or job['arm']=='pretrained'
    if not broad:challenges=[r for r in challenges if r['hidden_token_count']==3 and r['context_fraction']==1]
    if replay:
        selected=[]
        for k,f in sorted({(r['hidden_token_count'],r['context_fraction']) for r in challenges}):
            group=[r for r in challenges if r['hidden_token_count']==k and r['context_fraction']==f]
            selected.extend(group[i] for i in sorted({0,len(group)//3,2*len(group)//3,len(group)-1}))
        challenges=selected
    started=time.time();result={};budgets={}
    result['greedy'],budgets['greedy']=extract(m,challenges,t,1)
    beam=[r for r in challenges if r['hidden_token_count']==3 and r['context_fraction'] in c.CFG['beam_context_fractions']]
    if job['arm']=='single_flip':beam=[r for r in beam if r['context_fraction']==1]
    if broad or job['arm']=='single_flip':result['beam8'],budgets['beam8']=extract(m,beam,t,8)
    c.dump(out,result);c.dump(done,{'job':{k:str(v) if isinstance(v,Path) else v for k,v in job.items()},'seconds':time.time()-started,'budget':budgets,'prediction_sha256':c.sha(out),'FP32':True,'TF32':False,'truth_read_guard':True})
    c.event(event='conditional_attack_done',model=job['name'],seconds=time.time()-started,budgets=budgets)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--replay',action='store_true');parser.add_argument('--ready-only',action='store_true');parser.add_argument('--no-pretrained',action='store_true');args=parser.parse_args()
    c.fp32();t=c.tok();c.dump('attack_design_checks.json',check_algorithm())
    if not args.no_pretrained:
        m=c.load().eval().requires_grad_(False)
        for seed in c.CFG['seeds']:
            job={'seed':seed,'arm':'pretrained','epoch':0,'name':f'{seed}_pretrained'}
            run_job(job,m,t,'replay' if args.replay else 'predictions',args.replay)
        del m;gc.collect();torch.cuda.empty_cache()
    for job in c.checkpoint_jobs():
        if not job['path'].exists():
            if args.ready_only:continue
            raise FileNotFoundError(job['path'])
        if not args.replay and (c.OUT/'predictions'/f'{job["name"]}.DONE.json').exists():continue
        history=c.read(job['path'].parent/'history.json');record=next((r for r in history if r['epoch']==job['epoch']),None)
        if record is None:
            if args.ready_only:continue
            raise RuntimeError('Checkpoint has not been committed to training history: '+job['name'])
        assert c.sha(job['path'])==record['checkpoint_sha256']
        m=c.load(job['seed'],job['path']).eval().requires_grad_(False);run_job(job,m,t,'replay' if args.replay else 'predictions',args.replay)
        del m;gc.collect();torch.cuda.empty_cache()
    if not args.ready_only and not args.replay:c.dump('conditional_attack_DONE.json',{'unix':time.time(),'trained_checkpoints':33,'pretrained_datasets':5})
if __name__=='__main__':main()
