"""Separate evaluator. Never called by the training or blind generation program."""
import sys,itertools,collections,time,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import core as c
import numpy as np

def edit_distance(a,b):
    row=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        nxt=[i]
        for j,y in enumerate(b,1):nxt.append(min(nxt[-1]+1,row[j]+1,row[j-1]+(x!=y)))
        row=nxt
    return row[-1]

def conditional_records():
    records=[]
    for seed in c.CFG['seeds']:
        truth=c.read(c.OUT/f'private/truth_{seed}.json');data=c.read(c.OUT/f'private/data_{seed}.json')
        jobs=[{'seed':seed,'arm':'pretrained','epoch':0,'name':f'{seed}_pretrained'}]+[j for j in c.checkpoint_jobs() if j['seed']==seed]
        for job in jobs:
            path=c.OUT/'predictions'/f'{job["name"]}.json'
            if not path.exists():continue
            predictions=c.read(path)
            for method,outputs in predictions.items():
                for cid,p in outputs.items():
                    tr=truth[cid]
                    sides=('A','B') if job['arm']=='pretrained' else ('B' if job['arm']=='B' or (job['arm']=='single_flip' and tr['context_id']==data['single_flip_id']) else 'A',)
                    for side in sides:
                        target=tr[f'target_{side}'];twin=tr[f'target_{"B" if side=="A" else "A"}']
                        key={'seed':seed,'model':job['name'],'arm':job['arm'],'side':side,'epoch':job['epoch'],'method':method,'challenge_id':cid,'context_id':tr['context_id'],
                            'type':tr['type'],'repeat':tr['repeat'],'exposures':tr['repeat']*job['epoch'] if job['arm']!='pretrained' else 0,
                            'hidden_tokens':tr['hidden_tokens'],'context_fraction':tr['context_fraction'],'known_bp':tr['known_bp'],'hidden_bp':len(target),
                            'target_is_member':job['arm']!='pretrained' and tr['repeat']>0,'target':target,'twin':twin,'predicted':p['dna'],
                            'exact':int(p['dna']==target),'twin_exact':int(p['dna']==twin),'edit_similarity':1-edit_distance(target,p['dna'])/max(len(target),len(p['dna'])),
                            'token_accuracy':sum(x==y for x,y in zip(p['ids'],tr[f'target_ids_{side}']))/len(p['ids']),
                            'single_flip_target':job['arm']=='single_flip' and tr['context_id']==data['single_flip_id']}
                        records.append(key)
    return records

def summary(records):
    groups=collections.defaultdict(list)
    for r in records:
        category='pretrained' if r['arm']=='pretrained' else 'single_flip' if r['arm']=='single_flip' else 'trained'
        key=(category,r['epoch'],r['type'],r['repeat'],r['hidden_tokens'],r['context_fraction'],r['method'])
        groups[key].append(r)
    result=[]
    for key,rows in sorted(groups.items()):
        item=dict(zip(('category','epoch','type','repeat','hidden_tokens','context_fraction','method'),key))
        item.update(n=len(rows),hits=sum(r['exact'] for r in rows),rate=float(np.mean([r['exact'] for r in rows])),
            twin_hits=sum(r['twin_exact'] for r in rows),twin_rate=float(np.mean([r['twin_exact'] for r in rows])),
            advantage=float(np.mean([r['exact']-r['twin_exact'] for r in rows])),known_bp_range=[min(r['known_bp'] for r in rows),max(r['known_bp'] for r in rows)],
            hidden_bp_range=[min(r['hidden_bp'] for r in rows),max(r['hidden_bp'] for r in rows)],
            mean_edit_similarity=float(np.mean([r['edit_similarity'] for r in rows])),mean_token_accuracy=float(np.mean([r['token_accuracy'] for r in rows])),
            per_seed={str(seed):{'hits':sum(r['exact'] for r in rows if r['seed']==seed),'n':sum(r['seed']==seed for r in rows),
                               'twin_hits':sum(r['twin_exact'] for r in rows if r['seed']==seed)} for seed in c.CFG['seeds'] if any(r['seed']==seed for r in rows)})
        result.append(item)
    return result

def bootstrap(records):
    rng=np.random.default_rng(c.CFG['bootstrap_seed']);result=[]
    for kind in ('random','genomic','both'):
        for repeat in (1,4,16):
            rows=[r for r in records if r['arm'] in ('A','B') and r['epoch']==20 and r['method']=='greedy' and r['hidden_tokens']==3 and r['context_fraction']==1 and r['repeat']==repeat and (kind=='both' or r['type']==kind)]
            if not rows:continue
            arrays=[]
            for seed in c.CFG['seeds']:
                grouped=collections.defaultdict(list)
                for r in rows:
                    if r['seed']==seed:grouped[r['context_id']].append([r['exact'],r['twin_exact'],r['exact']-r['twin_exact']])
                if not grouped:continue
                assert all(len(vals)==2 for vals in grouped.values()),'A/B pair missing'
                arrays.append(np.array([np.mean(vals,axis=0) for vals in grouped.values()]))
            if len(arrays)!=5:continue
            draws=[]
            for _ in range(c.CFG['bootstrap_replicates']):
                samples=[]
                for i in rng.integers(0,len(arrays),len(arrays)):
                    a=arrays[int(i)];samples.append(a[rng.integers(0,len(a),len(a))].mean(0))
                draws.append(np.mean(samples,axis=0))
            ci=np.quantile(draws,[.025,.975],axis=0).T.tolist()
            result.append({'type':kind,'repeat':repeat,'n_model_target_evaluations':len(rows),'independent_training_seeds':5,
                'rate':float(np.mean([r['exact'] for r in rows])),'twin_rate':float(np.mean([r['twin_exact'] for r in rows])),
                'advantage':float(np.mean([r['exact']-r['twin_exact'] for r in rows])),'rate_ci95':ci[0],'twin_ci95':ci[1],'advantage_ci95':ci[2],
                'method':'seed bootstrap then context-pair bootstrap; A/B stay together; descriptive intervals'})
    return result

def flips(records):
    by={(r['model'],r['challenge_id'],r['method']):r for r in records};result=[]
    for seed in c.CFG['single_flip_seeds']:
        data=c.read(c.OUT/f'private/data_{seed}.json');cid=data['single_flip_id']+'_k3_c100'
        for method in ('greedy','beam8'):
            ka=(f'{seed}_A_e20',cid,method);kb=(f'{seed}_single_flip_e20',cid,method)
            if ka not in by or kb not in by:continue
            a=by[ka];b=by[kb];others=[r for r in records if r['model']==f'{seed}_single_flip_e20' and r['method']==method and not r['single_flip_target']]
            changed=sum(r['predicted']!=by[(f'{seed}_A_e20',r['challenge_id'],method)]['predicted'] for r in others)
            result.append({'seed':seed,'method':method,'context_id':data['single_flip_id'],'A_secret':a['target'],'B_secret':b['target'],
                'original_A_prediction':a['predicted'],'single_flip_prediction':b['predicted'],'A_recovered_before':bool(a['exact']),
                'B_recovered_after':bool(b['exact']),'A_recovered_after':bool(b['twin_exact']),'both_follow_training':bool(a['exact'] and b['exact']),
                'other_context_outputs_changed':changed,'other_context_n':len(others)})
    return result

def unconditional():
    rows=[];jobs=[{'name':'pretrained','arm':'pretrained','seeds':c.CFG['seeds']}]+[{'name':j['name'],'arm':j['arm'],'seeds':[j['seed']]} for j in c.checkpoint_jobs() if j['epoch']==20 and j['arm'] in ('A','B')]
    for job in jobs:
        path=c.OUT/'generations'/f'{job["name"]}.json'
        if not path.exists():continue
        generations=c.read(path);strings=[r['dna'] for r in generations];string_set=set(strings)
        for seed in job['seeds']:
            data=c.read(c.OUT/f'private/data_{seed}.json')
            for item in data['contexts']:
                for side in ('A','B'):
                    target=item[f'sequence_{side}'];matches=[i for i,s in enumerate(strings) if target in s]
                    rows.append({'model':job['name'],'seed':seed,'context_id':item['id'],'side':side,'type':item['type'],'repeat':item['repeat'],
                        'member':job['arm']==side and item['repeat']>0,'exact_output':int(target in string_set),'full_96bp_substring':int(bool(matches)),
                        'matching_generation_indices':matches,'generation_samples':len(strings)})
    groups=collections.defaultdict(list)
    for r in rows:groups[(r['model'],r['member'])].append(r)
    summaries=[{'model':k[0],'member':k[1],'targets':len(vals),'exact_outputs':sum(r['exact_output'] for r in vals),'full_96bp_substrings':sum(r['full_96bp_substring'] for r in vals),'generation_samples':vals[0]['generation_samples']} for k,vals in groups.items()]
    c.dump('unconditional_records.json',rows);return summaries

def markov(truth_records):
    output=[]
    for seed in c.CFG['seeds']:
        path=c.OUT/f'baselines/markov_{seed}.json'
        if not path.exists():continue
        p=c.read(path)
        for r in truth_records:
            if r['seed']==seed and r['arm'] in ('A','B') and r['epoch']==20 and r['method']=='greedy' and r['hidden_tokens']==3:
                output.append({**{k:r[k] for k in ('seed','type','repeat','context_fraction','side','challenge_id')},'exact':int(p[r['challenge_id']]==r['target']),'prediction':p[r['challenge_id']]})
    return output

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--partial',action='store_true');args=parser.parse_args()
    if not args.partial:
        assert (c.OUT/'conditional_attack_DONE.json').exists() and (c.OUT/'generation_DONE.json').exists()
    rows=conditional_records();c.dump('metrics_records.json',rows);s=summary(rows);c.dump('summary_metrics.json',s)
    if not args.partial:c.dump('primary_bootstrap.json',bootstrap(rows))
    c.dump('single_flip_results.json',flips(rows));c.dump('unconditional_summary.json',unconditional());c.dump('markov_records.json',markov(rows))
    headline=[r for r in s if r['category']=='trained' and r['epoch']==20 and r['method']=='greedy' and r['hidden_tokens']==3 and r['context_fraction']==1]
    c.event(event='analysis_complete',partial=args.partial,records=len(rows),headline=[{k:r[k] for k in ('type','repeat','n','hits','twin_hits')} for r in headline])
if __name__=='__main__':main()
