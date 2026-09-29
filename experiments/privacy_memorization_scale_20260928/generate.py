"""No-target-context generation: model, vocabulary, and fixed public budget only."""
import sys,time,gc,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import core as c
from attack import deny_truth
import numpy as np,torch

@torch.inference_mode()
def generate(m,t,n,seed):
    rng=np.random.default_rng(seed);lengths=rng.integers(c.CFG['generation_token_min'],c.CFG['generation_token_max']+1,n)
    generator=torch.Generator(device='cuda').manual_seed(seed);allowed=torch.tensor(c.dna_ids(t),device='cuda');outputs=[];m.float().eval()
    for start in range(0,n,32):
        ls=lengths[start:start+32];ids,att=c.pad([[t.mask_token_id]*int(length) for length in ls],t);batch=len(ls)
        for step in range(c.CFG['generation_steps']):
            hidden=m.bert(input_ids=ids,attention_mask=att)[0];logits=m.cls(hidden)[:,:,allowed]
            values,index=torch.topk(logits,k=c.CFG['generation_top_k'],dim=-1,sorted=True)
            probabilities=torch.softmax(values/c.CFG['generation_temperature'],-1)
            chosen=torch.multinomial(probabilities.reshape(-1,c.CFG['generation_top_k']),1,generator=generator).reshape(batch,64,1)
            token=allowed[index.gather(-1,chosen).squeeze(-1)];confidence=probabilities.gather(-1,chosen).squeeze(-1)
            confidence=confidence.masked_fill(ids.ne(t.mask_token_id),-float('inf'));rank=torch.argsort(confidence,dim=1,descending=True,stable=True)
            for i,length in enumerate(ls):
                count=min(int((ids[i]==t.mask_token_id).sum()),int(np.ceil(int(length)/c.CFG['generation_steps'])))
                if count:positions=rank[i,:count];ids[i,positions]=token[i,positions]
        assert not (ids==t.mask_token_id).any()
        for i,length in enumerate(ls):
            tokens=ids[i,1:1+int(length)].cpu().tolist();outputs.append({'index':start+i,'input_token_length':int(length),'ids':tokens,'dna':c.decode(t,tokens)})
    return outputs

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--replay',action='store_true');parser.add_argument('--ready-only',action='store_true');args=parser.parse_args();c.fp32();t=c.tok()
    jobs=[{'name':'pretrained','seed':42,'path':None}]+[j for j in c.checkpoint_jobs() if j['epoch']==20 and j['arm'] in ('A','B')]
    for job in jobs:
        if job['path'] is not None and not (job['path'].parent/'DONE.json').exists():
            if args.ready_only:continue
            raise FileNotFoundError(job['path'])
        folder=c.OUT/('generation_replay' if args.replay else 'generations');out=folder/f'{job["name"]}.json';done=out.with_suffix('.DONE.json')
        if done.exists():continue
        m=c.load(job['seed'],job['path']).eval().requires_grad_(False);started=time.time();n=32 if args.replay else c.CFG['generation_samples_per_model']
        rows=generate(m,t,n,c.CFG['generation_seed']);c.dump(out,rows)
        c.dump(done,{'unix':time.time(),'samples':len(rows),'seconds':time.time()-started,'generation_seed':c.CFG['generation_seed'],
            'target_context_provided':False,'target_specific_length_provided':False,'prediction_sha256':c.sha(out),'unique_strings':len({r['dna'] for r in rows})})
        c.event(event='unconditional_generation_done',model=job['name'],samples=len(rows),seconds=time.time()-started)
        del m;gc.collect();torch.cuda.empty_cache()
    if not args.replay and not args.ready_only:c.dump('generation_DONE.json',{'unix':time.time(),'models':11,'samples_per_model':c.CFG['generation_samples_per_model']})
if __name__=='__main__':main()
