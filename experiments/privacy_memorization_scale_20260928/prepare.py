import sys,time,platform,itertools
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import core as c
import numpy as np,pandas as pd,torch,transformers

def main():
    if (c.OUT/'preparation_DONE.json').exists():
        for name,digest in c.read(c.OUT/'preparation_DONE.json')['artifacts'].items():assert c.sha(c.OUT/name)==digest
        c.event(event='preparation_already_complete');return
    assert not (c.OUT/'preparation_started.json').exists(),'Inspect incomplete construction before restarting'
    c.dump('protocol_frozen.json',{'unix':time.time(),'protocol_sha256':c.sha(c.OUT/'protocol_KO.md'),'config_sha256':c.sha(c.OUT/'config.json'),
        'code_sha256':{p.name:c.sha(p) for p in c.OUT.glob('*.py')}})
    c.dump('preparation_started.json',{'unix':time.time()})
    t=c.tok();source=c.ROOT/'GUE_v2/EMP/H3K4me3/train.csv';devsource=c.ROOT/'GUE_v2/EMP/H3K4me3/dev.csv'
    train=pd.read_csv(source);dev=pd.read_csv(devsource);global_prefixes=set();global_canaries=set();audits=[];all_training_sequences=set()
    dependencies={str(p.relative_to(c.ROOT)):c.sha(p) for p in (source,devsource,c.SNAP/'pytorch_model.bin')}
    for seed in c.CFG['seeds']:
        rng=np.random.default_rng(seed+910000);order=rng.permutation(len(train));used_rows=set();background=[];bgcanon=set();cursor=0
        while len(background)<c.CFG['background_n']:
            index=int(order[cursor]);cursor+=1;s=str(train.iloc[index].sequence);s=s[(len(s)-96)//2:(len(s)-96)//2+96];key=c.canonical(s)
            if key in bgcanon:continue
            ids=t.encode(s,add_special_tokens=False);assert len(ids)+2<=64
            background.append({'source_row':index,'sequence':s,'ids':ids});used_rows.add(index);bgcanon.add(key)
        contexts=[];construction_rejections=0;attempts=[]
        for kind in c.CFG['context_types']:
            for repeat in c.CFG['repeats_per_epoch']:
                for j in range(c.CFG['contexts_per_type_repeat']):
                    while True:
                        if kind=='random':prefix=''.join(rng.choice(list('ACGT'),84));source_row=None;original_suffix=None
                        else:
                            source_row=int(order[cursor]);cursor+=1
                            if source_row in used_rows:continue
                            raw=str(train.iloc[source_row].sequence);crop=raw[(len(raw)-96)//2:(len(raw)-96)//2+96];prefix=crop[:84];original_suffix=crop[84:]
                        if prefix in global_prefixes:continue
                        pids=t.encode(prefix,add_special_tokens=False)
                        if not 12<=len(pids)<=59:construction_rejections+=1;continue
                        secrets=[];sids=[];trials=0
                        for attempt in range(1024):
                            secret=''.join(rng.choice(list('ACGT'),12));trials+=1
                            if secret in secrets or secret==original_suffix:continue
                            ids=t.encode(secret,add_special_tokens=False)
                            if len(ids)!=3 or t.encode(prefix+secret,add_special_tokens=False)!=pids+ids:continue
                            key=c.canonical(prefix+secret)
                            if key in bgcanon or key in global_canaries:continue
                            secrets.append(secret);sids.append(ids)
                            if len(secrets)==2:break
                        if len(secrets)==2:break
                        construction_rejections+=1
                    cid=rng.bytes(12).hex();global_prefixes.add(prefix);attempts.append(trials)
                    if source_row is not None:used_rows.add(source_row)
                    row={'id':cid,'type':kind,'repeat':repeat,'prefix':prefix,'prefix_ids':pids,'source_row':source_row,'reference_suffix':original_suffix,'construction_trials':trials}
                    for side,secret,ids in zip(('A','B'),secrets,sids):
                        row[f'secret_{side}']=secret;row[f'sequence_{side}']=prefix+secret;row[f'ids_{side}']=pids+ids
                        global_canaries.add(c.canonical(prefix+secret))
                    contexts.append(row)
        flip=next(r['id'] for r in contexts if r['type']=='genomic' and r['repeat']==16)
        data={'seed':seed,'background':background,'contexts':contexts,'single_flip_id':flip};c.dump(f'private/data_{seed}.json',data)
        challenges=[];truth={}
        for row in contexts:
            for k in c.CFG['hidden_bpe']:
                prefix_ids=row['ids_A'][:-k];assert prefix_ids==row['ids_B'][:-k]
                for fraction in c.CFG['context_fractions']:
                    n=int(len(prefix_ids)*fraction);known=prefix_ids[-n:] if n else [];cid=f'{row["id"]}_k{k}_c{int(fraction*100)}'
                    challenges.append({'id':cid,'prefix_ids':known,'hidden_token_count':k,'context_fraction':fraction})
                    truth[cid]={'context_id':row['id'],'type':row['type'],'repeat':row['repeat'],'hidden_tokens':k,'context_fraction':fraction,'known_bp':len(c.decode(t,known)),
                        **{f'target_{side}':c.decode(t,row[f'ids_{side}'][-k:]) for side in ('A','B')},
                        **{f'target_ids_{side}':row[f'ids_{side}'][-k:] for side in ('A','B')}}
        c.dump(f'public/challenges_{seed}.json',challenges);c.dump(f'private/truth_{seed}.json',truth)
        a=c.train_rows(data,'A');b=c.train_rows(data,'B');single=c.train_rows(data,'single_flip')
        assert all(len(x)==len(y)==len(z) for x,y,z in zip(a,b,single))
        assert sum(x!=z for x,z in zip(a,single))==16
        all_training_sequences.update(c.canonical(r['sequence']) for r in background)
        all_training_sequences.update(c.canonical(r[f'sequence_{side}']) for r in contexts for side in ('A','B'))
        audits.append({'seed':seed,'contexts':len(contexts),'sequences':len(contexts)*2,'epoch_examples':len(a),
            'background_n':len(background),'rejected_prefixes':construction_rejections,'secret_trial_quantiles':np.quantile(attempts,[0,.5,.9,1]).tolist(),
            'secret_gc_A':float(np.mean([(r['secret_A'].count('G')+r['secret_A'].count('C'))/12 for r in contexts])),
            'secret_gc_B':float(np.mean([(r['secret_B'].count('G')+r['secret_B'].count('C'))/12 for r in contexts])),
            'input_token_quantiles':np.quantile([len(r['ids_A']) for r in contexts],[0,.5,1]).tolist(),
            'A_B_identical_token_lengths':True,'single_flip_changed_epoch_slots':16,'A_B_same_attacker_inputs':True})
        c.event(event='dataset_constructed',**audits[-1])
    validation=[];excluded=0
    for index in np.random.default_rng(910999).permutation(len(dev)):
        s=str(dev.iloc[int(index)].sequence);s=s[(len(s)-96)//2:(len(s)-96)//2+96]
        if c.canonical(s) in all_training_sequences:excluded+=1;continue
        validation.append({'source_row':int(index),'sequence':s,'ids':t.encode(s,add_special_tokens=False)})
        if len(validation)==c.CFG['validation_n']:break
    assert len(validation)==512
    c.dump('private/validation.json',validation);c.dump('data_audit.json',{'seeds':audits,'validation_duplicate_exclusions':excluded,'unique_prefixes':len(global_prefixes),'unique_canary_canonical_sequences':len(global_canaries)})
    c.dump('public/vocab.json',{str(i):t.convert_ids_to_tokens(i) for i in c.dna_ids(t)})
    c.dump('environment.json',{'python':sys.version,'torch':torch.__version__,'transformers':transformers.__version__,'platform':platform.platform(),'gpu':torch.cuda.get_device_name(0),'dependencies':dependencies,'train_BF16':True,'eval_FP32':True})
    paths=[p for name in ('public','private') for p in (c.OUT/name).glob('*.json')]
    c.dump('preparation_DONE.json',{'unix':time.time(),'artifacts':{str(p.relative_to(c.OUT)):c.sha(p) for p in paths}})
if __name__=='__main__':main()
