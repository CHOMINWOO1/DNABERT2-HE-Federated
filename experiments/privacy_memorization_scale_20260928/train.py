import sys,time,gc,hashlib,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import core as c
import numpy as np,torch

@torch.inference_mode()
def validate(m,t,allowed):
    rows=c.read(c.OUT/'private/validation.json');generator=torch.Generator().manual_seed(912999);loss_sum=0.;n=0;correct=0
    m.eval()
    for start in range(0,len(rows),32):
        ids,att,y,count=c.mlm_batch([r['ids'] for r in rows[start:start+32]],t,generator,allowed)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            hidden=m.bert(input_ids=ids,attention_mask=att)[0];logits=m.cls(hidden[y!=-100]);labels=y[y!=-100]
            loss=torch.nn.functional.cross_entropy(logits.float(),labels,reduction='sum')
        loss_sum+=float(loss);n+=count;correct+=int((logits.argmax(-1)==labels).sum())
    return {'masked_ce':loss_sum/n,'masked_accuracy':correct/n,'masked_tokens':n,'precision':'BF16 forward / FP32 CE, fixed heldout public mask'}

def train(seed,arm,t,allowed):
    folder=c.model_dir(seed,arm)
    if (folder/'DONE.json').exists():
        done=c.read(folder/'DONE.json');assert c.sha(folder/'epoch_20.pt')==done['checkpoint_sha256'];return
    assert not (folder/'STARTED.json').exists(),'Incomplete training preserved: inspect before restarting'
    data=c.read(c.OUT/f'private/data_{seed}.json');rows=c.train_rows(data,arm);m=c.load(seed)
    opt=torch.optim.AdamW(m.parameters(),lr=c.CFG['lr'],weight_decay=c.CFG['weight_decay'])
    generator=torch.Generator().manual_seed(seed+991);rng=np.random.default_rng(seed+991);orders=[];history=[]
    folder.mkdir(parents=True,exist_ok=True);start=time.time();c.dump(folder/'STARTED.json',{'unix':start,'protocol_sha256':c.sha(c.OUT/'protocol_KO.md'),'seed':seed,'arm':arm})
    c.dump(folder/'initial_validation.json',validate(m,t,allowed))
    torch.cuda.reset_peak_memory_stats();steps=0
    for epoch in range(1,c.CFG['epochs']+1):
        m.train();order=rng.permutation(len(rows));orders.append(order);loss_sum=0.;targets=0
        for offset in range(0,len(order),32):
            ids,att,labels,count=c.mlm_batch([rows[int(i)] for i in order[offset:offset+32]],t,generator,allowed)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=m(input_ids=ids,attention_mask=att,labels=labels).loss
            if not torch.isfinite(loss):
                c.dump(folder/'FAILED.json',{'epoch':epoch,'step':steps,'reason':'nonfinite MLM loss'});raise RuntimeError('Nonfinite MLM loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();steps+=1
            loss_sum+=float(loss.detach())*count;targets+=count
        record={'epoch':epoch,'train_masked_ce':loss_sum/targets,'targets':targets,'steps':steps,'elapsed_seconds':time.time()-start}
        if epoch in c.CFG['checkpoint_epochs']:
            record['validation']=validate(m,t,allowed)
            state={n:p.detach().cpu().clone() for n,p in m.named_parameters()};torch.save(state,folder/f'epoch_{epoch:02d}.pt');del state
            record['checkpoint_sha256']=c.sha(folder/f'epoch_{epoch:02d}.pt')
        history.append(record);c.dump(folder/'history.json',history);c.event(event='epoch',seed=seed,arm=arm,**record)
    np.save(folder/'orders.npy',np.stack(orders))
    torch.save({'optimizer':opt.state_dict(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
        'mask_generator':generator.get_state(),'numpy_order_rng':rng.bit_generator.state,'steps':steps,'epoch':20},folder/'optimizer_rng.pt')
    c.dump(folder/'DONE.json',{'unix':time.time(),'elapsed_seconds':time.time()-start,'epochs':20,'steps':steps,'checkpoint_sha256':history[-1]['checkpoint_sha256'],
        'orders_sha256':c.sha(folder/'orders.npy'),'optimizer_rng_sha256':c.sha(folder/'optimizer_rng.pt'),'peak_gpu_bytes':torch.cuda.max_memory_allocated()})
    del opt,m;gc.collect();torch.cuda.empty_cache()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int);parser.add_argument('--arm');args=parser.parse_args()
    assert (c.OUT/'preparation_DONE.json').exists();t=c.tok();allowed=torch.tensor(c.dna_ids(t));torch.backends.cuda.matmul.allow_tf32=True
    for seed in ([args.seed] if args.seed else c.CFG['seeds']):
        arms=[args.arm] if args.arm else ['A','B']+(['single_flip'] if seed in c.CFG['single_flip_seeds'] else [])
        for arm in arms:train(seed,arm,t,allowed)
    if not args.seed and not args.arm:c.dump('training_DONE.json',{'unix':time.time(),'model_runs':13})
if __name__=='__main__':main()
