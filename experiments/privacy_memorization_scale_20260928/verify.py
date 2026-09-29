import sys,time,gc,argparse,collections
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import core as c
import numpy as np,torch

def data_check():
    t=c.tok();frozen=c.read(c.OUT/'protocol_frozen.json');assert c.sha(c.OUT/'protocol_KO.md')==frozen['protocol_sha256'];assert c.sha(c.OUT/'config.json')==frozen['config_sha256']
    for name,digest in frozen['code_sha256'].items():assert c.sha(c.OUT/name)==digest,name
    for name,digest in c.read(c.OUT/'environment.json')['dependencies'].items():assert c.sha(c.ROOT/name)==digest,name
    for name,digest in c.read(c.OUT/'preparation_DONE.json')['artifacts'].items():assert c.sha(c.OUT/name)==digest,name
    all_secrets=[];all_background=set();all_contexts=set();prefix_contains_secret=0;seen=[];checks=[]
    for seed in c.CFG['seeds']:
        data=c.read(c.OUT/f'private/data_{seed}.json');truth=c.read(c.OUT/f'private/truth_{seed}.json');challenges=c.read(c.OUT/f'public/challenges_{seed}.json')
        bg={c.canonical(r['sequence']) for r in data['background']};all_background|=bg
        for r in data['contexts']:
            assert r['prefix'] not in all_contexts;all_contexts.add(r['prefix'])
            for side in ('A','B'):
                full=r[f'sequence_{side}'];ids=r[f'ids_{side}'];assert len(full)==96 and len(r[f'secret_{side}'])==12
                assert t.encode(full,add_special_tokens=False)==ids and ids[:-3]==r['prefix_ids']
                assert c.decode(t,ids[-3:])==r[f'secret_{side}']
                all_secrets.append(c.canonical(full));prefix_contains_secret+=int(r[f'secret_{side}'] in r['prefix'])
            assert r['secret_A']!=r['secret_B'];assert r['secret_A']!=r['reference_suffix'] and r['secret_B']!=r['reference_suffix']
        byid={r['id']:r for r in data['contexts']}
        for p in challenges:
            tr=truth[p['id']];r=byid[tr['context_id']];k=p['hidden_token_count'];prefix=r['ids_A'][:-k];n=int(len(prefix)*p['context_fraction']);visible=prefix[-n:] if n else []
            assert visible==p['prefix_ids']
            for side in ('A','B'):assert c.decode(t,r[f'ids_{side}'][-k:])==tr[f'target_{side}']
            if k==3:assert tr['target_A']==r['secret_A'] and tr['target_B']==r['secret_B']
        a=c.train_rows(data,'A');b=c.train_rows(data,'B');flip=c.train_rows(data,'single_flip')
        assert all(len(x)==len(y)==len(z) for x,y,z in zip(a,b,flip));assert sum(x!=y for x,y in zip(a,flip))==16
        if seed in c.CFG['single_flip_seeds']:
            changed=[(x,y) for x,y in zip(a,flip) if x!=y];assert len({tuple(x) for x,y in changed})==1 and len({tuple(y) for x,y in changed})==1
        checks.append({'seed':seed,'same_A_B_attacker_inputs':True,'single_flip_unique_sequence_changed':1,'single_flip_epoch_slots_changed':16,'challenges':len(challenges)})
    assert len(all_secrets)==2560 and len(set(all_secrets))==2560
    assert not (set(all_secrets)&all_background)
    validation={c.canonical(r['sequence']) for r in c.read(c.OUT/'private/validation.json')};assert not validation&(all_background|set(all_secrets))
    c.dump('data_verification.json',{'unix':time.time(),'protocol_and_initial_code_unchanged':True,'source_dependencies_unchanged':True,'prepared_artifacts_unchanged':True,
        'unique_canary_sequences':len(set(all_secrets)),'unique_contexts':len(all_contexts),'canary_background_exact_rc_overlap':0,'validation_training_exact_rc_overlap':0,
        'full_secret_coincidentally_present_in_known_prefix':prefix_contains_secret,'per_seed':checks,'homology_screen':'No exhaustive homology scan; genomic twins deliberately share public context'})

def training_replay():
    results=[];t=c.tok();allowed=torch.tensor(c.dna_ids(t));torch.backends.cuda.matmul.allow_tf32=True
    for arm in ('A','B'):
        seed=42;data=c.read(c.OUT/f'private/data_{seed}.json');rows=c.train_rows(data,arm);m=c.load(seed);m.train()
        opt=torch.optim.AdamW(m.parameters(),lr=c.CFG['lr'],weight_decay=c.CFG['weight_decay']);gen=torch.Generator().manual_seed(seed+991);order=np.random.default_rng(seed+991).permutation(len(rows))
        for offset in range(0,len(rows),32):
            ids,att,labels,count=c.mlm_batch([rows[int(i)] for i in order[offset:offset+32]],t,gen,allowed);opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=m(input_ids=ids,attention_mask=att,labels=labels).loss
            loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
        saved=torch.load(c.model_dir(seed,arm)/'epoch_01.pt',map_location='cpu',weights_only=True);maximum=0.;equal=True
        for name,p in m.named_parameters():
            current=p.detach().cpu();equal=equal and torch.equal(current,saved[name]);maximum=max(maximum,float((current-saved[name]).abs().max()))
        result={'seed':seed,'arm':arm,'epoch':1,'steps':106,'bitwise_equal':equal,'max_abs_parameter_error':maximum};results.append(result);c.event(event='training_replay',**result)
        assert equal
        del m,opt,saved;gc.collect();torch.cuda.empty_cache()
    c.dump('training_replay.json',results)

def final_check():
    assert (c.OUT/'training_DONE.json').exists() and (c.OUT/'conditional_attack_DONE.json').exists() and (c.OUT/'generation_DONE.json').exists()
    checkpoints=0;steps=0;models=0
    for seed in c.CFG['seeds']:
        orders=None;targets=None
        for arm in ['A','B']+(['single_flip'] if seed in c.CFG['single_flip_seeds'] else []):
            folder=c.model_dir(seed,arm);done=c.read(folder/'DONE.json');history=c.read(folder/'history.json')
            assert len(history)==20 and done['steps']==2120
            for row in history:
                if 'checkpoint_sha256' in row:
                    assert c.sha(folder/f'epoch_{row["epoch"]:02d}.pt')==row['checkpoint_sha256'];checkpoints+=1
            assert c.sha(folder/'orders.npy')==done['orders_sha256'] and c.sha(folder/'optimizer_rng.pt')==done['optimizer_rng_sha256']
            current=np.load(folder/'orders.npy');assert current.shape==(20,3392)
            if orders is not None:assert np.array_equal(orders,current) and targets==[r['targets'] for r in history]
            orders=current;targets=[r['targets'] for r in history];steps+=done['steps'];models+=1
    replay_count=0;max_score_error=0.
    for path in (c.OUT/'replay').glob('*.json'):
        if path.name.endswith('.DONE.json'):continue
        expected=c.read(c.OUT/'predictions'/path.name)
        for method,rows in c.read(path).items():
            for cid,p in rows.items():
                old=expected[method][cid];assert p['ids']==old['ids'] and p['top_ties']==old['top_ties']
                max_score_error=max(max_score_error,abs(p['selected_log_score']-old['selected_log_score']));replay_count+=1
    assert replay_count==944
    generation_replayed=0
    for path in (c.OUT/'generation_replay').glob('*.json'):
        if path.name.endswith('.DONE.json'):continue
        rows=c.read(path);expected=c.read(c.OUT/'generations'/path.name);assert rows==expected[:len(rows)];generation_replayed+=len(rows)
    assert generation_replayed==11*32
    # Recheck checkpoint bindings for every attack artifact and complete grid sizes.
    jobs=[{'name':f'{seed}_pretrained','seed':seed,'arm':'pretrained','epoch':0} for seed in c.CFG['seeds']]+c.checkpoint_jobs()
    for job in jobs:
        p=c.OUT/'predictions'/f'{job["name"]}.json';done=c.read(p.with_suffix('.DONE.json'));assert c.sha(p)==done['prediction_sha256']
        rows=c.read(p);broad=job['arm']=='pretrained' or (job['epoch']==20 and job['arm'] in ('A','B'))
        assert len(rows['greedy'])==(3072 if broad else 256)
        if broad:assert len(rows['beam8'])==512
        elif job['arm']=='single_flip':assert len(rows['beam8'])==256
    for p in (c.OUT/'generations').glob('*.DONE.json'):
        done=c.read(p);prediction=p.with_name(p.name.replace('.DONE.json','.json'));assert c.sha(prediction)==done['prediction_sha256'] and done['samples']==2048
    c.dump('verification.json',{'unix':time.time(),'models':models,'checkpoints':checkpoints,'optimizer_steps_total':steps,
        'data_audit':c.read(c.OUT/'data_verification.json'),'A_B_flip_orders_identical':True,'A_B_flip_mask_target_counts_identical':True,
        'training_replay':c.read(c.OUT/'training_replay.json'),'conditional_replay_predictions':replay_count,'conditional_max_log_score_difference':max_score_error,
        'unconditional_replay_exact_samples':generation_replayed,'all_attack_grid_sizes_verified':True,'truth_read_guard_used':True})
    c.event(event='verification_complete',models=models,checkpoints=checkpoints,replay=replay_count,max_score_difference=max_score_error)

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-only',action='store_true');p.add_argument('--training-replay-only',action='store_true');args=p.parse_args()
    if args.training_replay_only:training_replay();return
    data_check()
    if not args.data_only:final_check()
if __name__=='__main__':main()
