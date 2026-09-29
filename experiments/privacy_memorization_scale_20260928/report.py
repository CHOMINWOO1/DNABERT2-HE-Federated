import sys,time,collections
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
sys.path.insert(0,str(Path(__file__).resolve().parent))
import core as c
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    summary=c.read(c.OUT/'summary_metrics.json');records=c.read(c.OUT/'metrics_records.json');primary=c.read(c.OUT/'primary_bootstrap.json')
    flips=c.read(c.OUT/'single_flip_results.json');unconditional=c.read(c.OUT/'unconditional_summary.json');verify=c.read(c.OUT/'verification.json');data_audit=c.read(c.OUT/'data_audit.json')
    assert verify['models']==13 and verify['all_attack_grid_sizes_verified']
    def get(kind,repeat,k=3,fraction=1.,method='greedy',epoch=20,category='trained'):
        return next(r for r in summary if (r['category'],r['epoch'],r['type'],r['repeat'],r['hidden_tokens'],r['context_fraction'],r['method'])==(category,epoch,kind,repeat,k,fraction,method))
    def combine(rows):return sum(r['hits'] for r in rows),sum(r['n'] for r in rows)
    def pct(hits,n):return str((Decimal(hits)*100/Decimal(n)).quantize(Decimal('0.01'),rounding=ROUND_HALF_UP))
    kinds=['random','genomic'];colors={'random':'#277DA1','genomic':'#23856D'};names={'random':'Random context','genomic':'Genomic context'}
    figures=c.OUT/'figures';figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    def save(fig,name):
        for suffix in ('png','pdf','svg'):fig.savefig(figures/f'{name}.{suffix}',dpi=200,bbox_inches='tight')
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.4,4.7))
    for kind in kinds:
        rates=[100*get(kind,r)['rate'] for r in (0,1,4,16)];ax.plot(range(4),rates,'o-',label=names[kind],color=colors[kind],lw=2)
        for j,r in enumerate((1,4,16),1):
            b=next(x for x in primary if x['type']==kind and x['repeat']==r);lo,hi=np.array(b['rate_ci95'])*100
            ax.errorbar(j,100*b['rate'],yerr=[[max(0,100*b['rate']-lo)],[max(0,hi-100*b['rate'])]],fmt='none',color=colors[kind],capsize=4)
    ax.axhline(0,color='#777777',lw=.8,ls=':');ax.set_xticks(range(4),['0','20','80','320']);ax.set_xlabel('Exposure group (total training presentations per sequence)');ax.set_ylabel('Exact hidden 12 bp recovery (%)');ax.set_ylim(-3,104)
    ax.set_title('Final-weight extraction depends on training exposure');ax.legend(frameon=False,loc='upper left');ax.grid(axis='y',alpha=.15)
    fig.text(.12,.005,'5 seeds; paired A/B secrets; 84 bp context; greedy. Error bars: hierarchical bootstrap 95% CI.',fontsize=8);fig.tight_layout(rect=(0,.03,1,1));save(fig,'figure1_exposure')
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),sharey=True)
    for ax,kind in zip(axes,kinds):
        array=np.array([[100*get(kind,16,k,f)['rate'] for f in (1.,.5,.25,0.)] for k in (3,6,12)])
        ax.imshow(array,vmin=0,vmax=100,cmap='YlGnBu',aspect='auto');ax.set_title(names[kind]);ax.set_xticks(range(4),['100%','50%','25%','0%']);ax.set_xlabel('Retained prefix tokens supplied')
        ax.set_yticks(range(3),['3 BPE (12 bp)','6 BPE','12 BPE'])
        for i in range(3):
            for j in range(4):ax.text(j,i,f'{array[i,j]:.1f}%',ha='center',va='center',color='white' if array[i,j]>55 else '#222222')
    axes[0].set_ylabel('Hidden suffix length');fig.suptitle('Context and hidden-length dependence (320 presentations)',fontsize=13)
    fig.text(.03,.015,'0% still supplies the number of hidden tokens. This panel is not whole-sequence generation.',fontsize=9);fig.tight_layout(rect=(0,.05,1,.93));save(fig,'figure2_context_length')
    fig,axes=plt.subplots(1,2,figsize=(10,4.4))
    for kind in kinds:
        vals=[100*get(kind,16,epoch=epoch)['rate'] for epoch in (5,10,20)];axes[0].plot([5,10,20],vals,'o-',color=colors[kind],label=names[kind])
    axes[0].set(xlabel='Training epoch',ylabel='Exact hidden 12 bp recovery (%)',ylim=(-3,104));axes[0].set_title('16 presentations per epoch');axes[0].set_xticks([5,10,20]);axes[0].legend(frameon=False)
    vals=[]
    for seed in c.CFG['seeds']:
        for arm in ('A','B'):
            folder=c.model_dir(seed,arm);hist=c.read(folder/'history.json');v=[c.read(folder/'initial_validation.json')['masked_ce']]+[r['validation']['masked_ce'] for r in hist if r['epoch'] in (1,5,10,20)]
            vals.append(v);axes[1].plot([0,1,5,10,20],v,alpha=.25,color='#576575',lw=1)
    axes[1].plot([0,1,5,10,20],np.mean(vals,axis=0),color='#23364D',lw=2,label='Mean of 10 models');axes[1].set(xlabel='Training epoch',ylabel='Public validation masked CE');axes[1].set_xticks([0,5,10,20]);axes[1].set_title('Fixed public validation masks');axes[1].legend(frameon=False)
    fig.tight_layout();save(fig,'figure3_training_and_validation')
    rows=[]
    for repeat in (0,1,4,16):
        row=[str(repeat*20)]
        for kind in kinds:
            r=get(kind,repeat);row.append(f'{r["hits"]}/{r["n"]} ({pct(r["hits"],r["n"])}%)');row.append(f'{r["twin_hits"]}/{r["n"]}')
        rows.append('| '+' | '.join(row)+' |')
    dose_table='| 총 노출 | 무작위 문맥: 학습 비밀 | 짝 비밀 | genomic 문맥: 학습 비밀 | 짝 비밀 |\n|---:|---:|---:|---:|---:|\n'+'\n'.join(rows)
    grid=[]
    for kind in kinds:
        for k in (3,6,12):
            row=[kind,str(k),f'{get(kind,16,k)["hidden_bp_range"][0]}–{get(kind,16,k)["hidden_bp_range"][1]}']
            row.extend(f'{get(kind,16,k,f)["hits"]}/{get(kind,16,k,f)["n"]} ({pct(get(kind,16,k,f)["hits"],get(kind,16,k,f)["n"])}%)' for f in (1.,.5,.25,0.))
            grid.append('| '+' | '.join(row)+' |')
    grid_table='| 문맥 종류 | 숨긴 BPE | 숨긴 bp 범위 | 전체 문맥 | 절반 | 1/4 | 없음 |\n|---|---:|---:|---:|---:|---:|---:|\n'+'\n'.join(grid)
    seed_table=[]
    for seed in c.CFG['seeds']:
        a=get('random',16)['per_seed'][str(seed)];b=get('genomic',16)['per_seed'][str(seed)];seed_table.append(f'| {seed} | {a["hits"]}/{a["n"]} | {b["hits"]}/{b["n"]} |')
    flip_table=[]
    for r in flips:flip_table.append(f'| {r["seed"]} | {r["method"]} | `{r["A_secret"]}` | `{r["B_secret"]}` | `{r["original_A_prediction"]}` | `{r["single_flip_prediction"]}` | {r["other_context_outputs_changed"]}/{r["other_context_n"]} |')
    primary_all=next(r for r in primary if r['type']=='both' and r['repeat']==16);high_n=primary_all['n_model_target_evaluations'];high_hits=sum(get(k,16)['hits'] for k in kinds)
    ci=primary_all['rate_ci95'];adv_ci=primary_all['advantage_ci95'];beam_hits,beam_n=combine([get(k,16,method='beam8') for k in kinds])
    free_members=[r for r in unconditional if r['member']];free_n=sum(r['targets'] for r in free_members);free_hits=sum(r['full_96bp_substrings'] for r in free_members);free_exact=sum(r['exact_outputs'] for r in free_members)
    free_table='| 모델 | 대상 학습 여부 | target 평가 수 | 출력 전체 일치 | 전체96 bp 포함 | 생성 수 |\n|---|---|---:|---:|---:|---:|\n'+'\n'.join(f'| {r["model"]} | {"학습" if r["member"] else "미학습"} | {r["targets"]} | {r["exact_outputs"]} | {r["full_96bp_substrings"]} | {r["generation_samples"]} |' for r in unconditional)
    prior_hits,prior_n=combine([get(k,16,epoch=0,category='pretrained') for k in kinds])
    dose20=combine([get(k,1) for k in kinds]);dose80=combine([get(k,4) for k in kinds])
    half=combine([get(k,16,fraction=.5) for k in kinds]);quarter=combine([get(k,16,fraction=.25) for k in kinds])
    longer=combine([get(k,16,k=12) for k in kinds])
    longer_bp=[min(get(k,16,k=12)['hidden_bp_range'][0] for k in kinds),max(get(k,16,k=12)['hidden_bp_range'][1] for k in kinds)]
    flip_g=[r for r in flips if r['method']=='greedy'];flip_b=[r for r in flips if r['method']=='beam8']
    markov=c.read(c.OUT/'markov_records.json');markhigh=[r for r in markov if r['repeat']==16 and r['context_fraction']==1]
    bgCE=float(np.mean([c.read(c.model_dir(seed,arm)/'history.json')[-1]['validation']['masked_ce'] for seed in c.CFG['seeds'] for arm in ('A','B')]))
    allval=np.array(vals);baseCE=float(allval[:,0].mean())
    manuscript=f'''# 원고용 방법·결과 초안

## 연구 범위

우리는 DNABERT-2의 최종 가중치에서 학습된 합성 DNA를 조건부로 추출할 수 있는지 평가하였다. 일반 MLM 미세조정을 사용했으며, 공격에 gradient나 학습 중 업데이트를 제공하지 않았다. 본 연구는 공개 효모 유래 문맥과 무작위 문맥을 사용한 통제 실험이다. 실제 환자 유전체나 임상 변이를 평가하지 않았다. GUE의 epigenetic marks prediction 자료가 효모를 대상으로 한다는 설명은 [DNABERT-2 논문 Appendix B.2](https://arxiv.org/html/2306.15006v2)에 제시되어 있다.

## 방법

각 seed에서 공개 H3K4me3 서열의 중앙 96 bp 구간 2,048개를 배경 자료로 사용하였다. 암기 여부를 확인하기 위해 삽입하는 평가 서열(canary)은 84 bp 공통 문맥과 12 bp 합성 비밀을 결합해 만들었다. 무작위 문맥과 공개 유전체 유래 문맥 각각에 대해 epoch당 0, 1, 4, 16회 노출 조건을 구성하였다. 각 조건에는 32개 문맥이 포함되었다. 동일 문맥에 서로 다른 비밀 A와 B를 배정하여 seed당 256개 문맥과 512개 서열을 생성하였다. 5개 seed 전체에는 1,280개 문맥과 2,560개 서열이 포함되었다.

비밀은 정확히 3개 BPE token으로 표현되는 12 bp 문자열로 제한하였다. 전체 서열의 토큰화가 prefix와 secret의 개별 토큰화를 연결한 결과와 같은지 확인하였다. 이 조건은 제공한 prefix에 비밀의 일부가 섞이는 것을 방지하지만, 생성된 비밀 분포를 제한한다. 두 비밀은 원래 공개 서열의 접미사와 다르며, A/B는 동일한 공격 입력을 공유한다.

A 모델에는 A만, B 모델에는 B만 노출하였다. 두 모델의 배경 자료, 노출 횟수, token 길이, 예시 순서와 masking 난수를 맞췄다. 추가로 3개 seed에서 높은 노출의 genomic 문맥 하나만 A에서 B로 교체하였다. 이 대조에서는 다른 서열과 순서를 유지하였다. 그룹 단위 교체의 10개 모델과 단일 문맥 교체의 3개 모델을 합해 13개 모델을 학습하였다.

모든 모델은 동일한 공개 DNABERT-2 checkpoint에서 시작하였다. AdamW, 학습률 5e-5, weight decay 0.01, batch size 32와 20 epochs를 사용하였다. 일반 dynamic MLM에서 15%의 token을 선택하고, 선택 위치의 80%는 MASK, 10%는 무작위 DNA token, 10%는 원래 token으로 유지하였다. BF16으로 학습하고 gradient norm을 1로 제한하였다. 각 epoch는 3,392개 예시를 포함하였다. 마지막 checkpoint를 주 평가 대상으로 고정하고, 공격 결과로 checkpoint를 선택하지 않았다.

주 공격은 최종 모델과 84 bp prefix, 숨긴 3-token 길이를 입력받아 greedy로 12 bp secret을 생성하였다. 정답이나 정답 후보 목록은 제공하지 않았다. 정확 복구는 숨긴 문자열 전체 일치로 정의하였다. 남은 prefix token의 100%, 50%, 25%, 0%를 제공하는 조건과 숨긴 3, 6, 12 BPE를 비교하였다. Beam width 8의 탐색은 숨긴 3 BPE와 전체/절반 문맥에서 보조 분석으로 사용하였다. 모든 공격은 FP32로 실행하고 TF32와 autocast를 해제하였다.

문맥 없는 추출에서는 목표 서열이나 목표별 길이를 제공하지 않았다. 고정 예산으로 각 최종 A/B 모델에서 2,048개 문자열을 생성하였다. 공개 범위 12–40 token의 MASK 입력에서 시작해 12단계 confidence-first sampling을 수행하였다. 생성 문자열이 학습 canary 96 bp 전체와 일치하는 경우와 전체 canary를 연속 부분문자열로 포함하는 경우를 구분하였다.

주 불확실성 분석은 seed를 재표집한 뒤 해당 seed 안에서 문맥 pair를 재표집하는 5,000회 hierarchical bootstrap이었다. A/B pair를 함께 유지하였다. 5개 seed의 개별 결과를 제시하며, 구간은 통제 실험의 기술적 불확실성으로 해석한다. 여러 문맥·길이 조건은 탐색 분석이고 각각을 독립 확증 검정으로 취급하지 않았다.

성공이 한 건도 없는 그룹의 재표집 구간은 [0, 0]으로 퇴화한다. 이러한 구간을 모집단 추출 확률이 0이라는 근거나 위험의 상한으로 사용하지 않았다.

## 결과

총 320회 노출한 서열의 12 bp secret은 전체 84 bp 문맥에서 greedy로 {high_hits}/{high_n}개({100*primary_all['rate']:.2f}%) 복구되었다. Hierarchical bootstrap 95% 구간은 {100*ci[0]:.2f}–{100*ci[1]:.2f}%였다. 동일 문맥의 미학습 twin 복구율은 {100*primary_all['twin_rate']:.2f}%였으며, 학습·미학습 twin 차이의 95% 구간은 {100*adv_ci[0]:.2f}–{100*adv_ci[1]:.2f}%p였다. 사전학습 모델의 같은 target 평가에서는 {prior_hits}/{prior_n}개가 일치하였다. Beam width 8 보조 분석의 정확 복구는 {beam_hits}/{beam_n}개였다. 두 공격에서 성능이 좋은 쪽을 주 결과로 선택하지 않았다.

전체 문맥을 제공했을 때 총 20회 노출 그룹의 복구는 {dose20[0]}/{dose20[1]}개였고, 80회 노출 그룹은 {dose80[0]}/{dose80[1]}개({pct(*dose80)}%)였다. 이는 서로 다른 문맥 그룹의 비교이며, 동일 서열을 모든 노출량에서 재학습한 인과효과는 아니다.

320회 노출 그룹에서 제공하는 prefix token을 절반으로 줄이면 12 bp 복구는 {half[0]}/{half[1]}개({pct(*half)}%), 1/4로 줄이면 {quarter[0]}/{quarter[1]}개({pct(*quarter)}%)였다. 숨긴 구간을 12 BPE로 늘린 조건에서는 남은 문맥 전체를 제공했을 때 {longer[0]}/{longer[1]}개({pct(*longer)}%)가 정확히 일치하였다. 이때 숨긴 구간은 {longer_bp[0]}–{longer_bp[1]} bp로, 합성 비밀 12 bp와 원래 공통 prefix의 일부를 함께 포함한다. 이러한 문맥·길이 분석은 사전 지정한 탐색 결과다.

단일 문맥 교체 대조에서는 greedy가 {sum(r['both_follow_training'] for r in flip_g)}/{len(flip_g)}건, beam width 8이 {sum(r['both_follow_training'] for r in flip_b)}/{len(flip_b)}건에서 교체 전 A와 교체 후 B를 모두 복구하였다. 이는 3개 대상에 대한 사례 대조이며 모집단 복구율의 추정치가 아니다. 교체 대상 이외의 문맥에서 발생한 출력 변화도 함께 기록하였다.

문맥 없는 생성에서는 10개 미세조정 모델의 학습 canary 평가 {free_n}건 중 전체 96 bp가 생성 문자열에 포함된 경우는 {free_hits}건이었고, 출력 전체와 정확히 일치한 경우는 {free_exact}건이었다. 생성 예산은 모델당 2,048개였다. 같은 예산의 사전학습 기준선과 미학습 서열 대조는 별도 표에 제시한다. 제한된 생성 절차의 실패를 무문맥 추출의 불가능성으로 해석하지 않는다.

## 해석과 한계

관측된 추출 결과와 A/B 대조는 알려진 문맥을 이용한 출력이 학습에 노출된 개별 합성 내용에 의존할 수 있다는 근거를 제공한다. 제시된 성공률은 조건부 비밀 추출률이며, 제공된 84 bp를 새로 복원한 것으로 세지 않았다. 서열 예측 목적, 반복 노출량, 공개 문맥과 탐색 예산을 명시해야 한다.

하나의 backbone, 공개 효모 자료의 짧은 crop, 조건부로 생성한 합성 secret과 높은 반복 노출을 사용하였다. 높은 노출 canary가 전체 학습 예시의 상당 비중을 차지하는 stress 조건이다. 실제 민감 변이·인간 유전체·다른 foundation model에서의 추출률을 측정하지 않았다. 가까운 상동성의 전수 배제도 수행하지 않았다. canary 기반 암기 검증은 [The Secret Sharer](https://www.usenix.org/conference/usenixsecurity19/presentation/carlini)와 연결되며, 유전체 모델에 대한 [기존 암기 연구](https://arxiv.org/html/2603.08913v1)가 존재한다. 신규성이 확정되었다거나 최초의 유전체 암기 발견이라고 주장하지 않는다.

수치 출처: primary_bootstrap.json, summary_metrics.json, single_flip_results.json, unconditional_summary.json. 작성 과정에서는 수치와 인용을 바꾸지 않고, 일반적인 유전체 복원 주장을 실험 조건의 범위로 구체화하였다.
'''
    (c.OUT/'manuscript_methods_results_KO.md').write_text(manuscript,encoding='utf-8')
    report=f'''# 최종 가중치 암기·복구 확장 실험 결과

2026-09-28. 완료된 통제 실험: 5 seed, 1,280개 문맥, 2,560개 합성 서열, 13개 MLM FullFT 실행, checkpoint52개. 연합학습·gradient·HE는 이번 연구 범위에 포함하지 않았다.

## 가장 중요한 결과

총320회 학습에 노출한 비밀의 조건부 정확 복구는 **{high_hits}/{high_n} ({100*primary_all['rate']:.2f}%)**였다. 모델은84 bp prefix와 hidden3 BPE를 입력받아 숨긴12 bp를 생성했다. 학습하지 않은 같은-prefix twin의 복구율은 **{100*primary_all['twin_rate']:.2f}%**였고, 사전학습 기준선은 **{prior_hits}/{prior_n}**였다. 이 결과는 최종 가중치에 남은 학습 정보의 조건부 추출을 뒷받침한다. 실제 환자 정보나 아무 문맥 없이 전체 유전체를 복원했다는 결과로 표현하지 않는다.

Greedy가 사전 고정한 주 공격이다. beam8 보조 공격에서는 {beam_hits}/{beam_n}개였다. 통합 greedy95% 구간은 {100*ci[0]:.2f}–{100*ci[1]:.2f}%이며 seed와 A/B문맥 pair를 묶은 hierarchical bootstrap으로 구했다. 이 구간은5개 seed와 합성 평가 분포의 불확실성이며 임상 유출률의 신뢰구간이 아니다.

성공 0건인 그룹의 bootstrap [0, 0] 구간은 관측값 재표집의 한계다. 이를 추출 확률이 0이라는 증거나 일반적인 위험 상한으로 해석하지 않는다. 이번 모델은 DNA 빈칸 예측을 학습하는 MLM FullFT 모델이며, 이전 H3K4me3 분류 목적의 FullFT 결과와 구분한다.

## 노출량

{dose_table}

각 칸의 분모320은5 seed×32문맥×2개 학습 arm의 model-target 평가이다. A/B는 같은 문맥을 공유하므로320개를 모두 독립 모델처럼 취급하지 않는다. 0회 그룹은 두 secret 모두 미학습이며 표에서 A/B의 배정 target을 편의상 비교한 것이다. 전체 입력96 bp 중 제공한84 bp는 복구 성과에 포함하지 않았다.

![노출량]({(figures/'figure1_exposure.png').as_posix()})

| seed | 무작위 문맥,320회 | genomic 문맥,320회 |
|---:|---:|---:|
{chr(10).join(seed_table)}

공개 DNA의2차 Markov 기준선은 주320회/전체문맥 조건에서 {sum(r['exact'] for r in markhigh)}/{len(markhigh)}개였다. 이것은12 bp 길이를 아는 기준선이고, BPE 수로 길이를 지정하는 모델 공격과의 정보 차이를 보고서에 명시한다. 원래 사전학습 모델과 untrained twin을 주 비교로 사용한다.

## 문맥 지식과 비밀 길이

아래는320회 노출, greedy 기준이다. 문맥 비율은 숨길 부분을 제외한 prefix의 **token 수**를 기준으로 한다. 주 secret3 BPE는12 bp로 고정됐고,6/12 BPE는 공통 prefix 일부까지 가리므로 bp 길이가 달라진다.

{grid_table}

![문맥과 길이]({(figures/'figure2_context_length.png').as_posix()})

0% 문맥에도 hidden token 수는 주어진다. 이 표의 짧은 접미사 생성과 아래의 전체96 bp 무문맥 생성은 다른 실험이다. 기존 token 경계를 공격자가 안다는 가정도 남아 있다.

## 동일 문맥의 학습 secret 교환

A/B 주 실험은 모든 노출 canary의 비밀을 함께 교환하므로 그룹 counterfactual이다. 별도의 single_flip에서는 각 seed의 사전에 지정한 genomic·repeat16 문맥 하나만 A→B로 바꾸었다. 한 epoch의16개 반복 슬롯을 제외한 입력은 동일하고, 예시 순서·token 길이·mask 수가 일치했다.

| seed | 공격 | A 정답 | B 정답 | A 모델 출력 | 한 비밀 교체 후 출력 | 다른 문맥의 출력 변화 |
|---:|---|---|---|---|---|---:|
{chr(10).join(flip_table)}

Greedy에서 교체 전후 모두 해당 학습 비밀을 복구한 사례는 {sum(r['both_follow_training'] for r in flip_g)}/{len(flip_g)}, beam8은 {sum(r['both_follow_training'] for r in flip_b)}/{len(flip_b)}이다. 성공하지 않은 사례도 그대로 보존했다. 교체 후 다른 학습의 경로가 달라질 수 있으므로 모든 가중치가 단일 위치에서만 달라진다고 주장하지 않는다.

## 문맥 없는 전체 서열 생성

10개 최종 A/B 모델 각각2,048개, 사전학습 모델2,048개로 총22,528개 문자열을 생성했다. 목표별 prefix·길이·ID는 제공하지 않았다. 모델별 입력 길이는 공개 고정 seed로12–40 token에서 선택했고, all-MASK에서12단계 top50 sampling으로 채웠다.

학습 canary의 model-target 평가 {free_n}건 중 **전체96 bp 연속 substring 발견 {free_hits}건**, **생성 출력 전체와의 정확 일치 {free_exact}건**이었다. 원예측은 generations/에 보존했다. 한두 짧은 조각의 출현이나 공개 prefix의 일치를 전체 서열 복구로 계산하지 않았다. 전체96 bp 추출이 실패한 경우에도 다른 생성 공격과 더 큰 예산까지 불가능하다는 의미는 아니다.

{free_table}

사전학습 모델의2,048개 생성물을5개 seed 데이터 전체에 재사용했다. 동일한 생성물을 target마다 비교한 것이므로 target 수를 독립 생성 시행 수로 취급하지 않는다.

## 학습과 공개 자료 성능

전체 모델은20 epochs×106 optimizer steps=2,120 steps로 학습했다. 공개 검증512개에 동일 mask를 사용한 masked CE는 초기 평균 {baseCE:.4f}, 최종 주10개 모델 평균 {bgCE:.4f}였다. 이 지표는 MLM 성능이고 H3K4me3 분류 AUROC가 아니다. 공격 결과로 최적 checkpoint를 선택하지 않았다.

![학습 시점]({(figures/'figure3_training_and_validation.png').as_posix()})

## 데이터·주장 범위

공개 background는 GUE H3K4me3의 효모 서열 중앙96 bp crop이다. 종은 [DNABERT-2 논문 Appendix B.2](https://arxiv.org/html/2306.15006v2)에서 확인했다. genomic canary도 공개 효모 문맥84 bp에 무작위12 bp를 넣은 합성 자료이며 인간의 SNP나 환자 식별 자료가 아니다.

각 모델에서 배경2,048개와 노출 canary192개를 사용했다. 반복을 포함한 예시3,392개 중 canary는1,344개(39.62%)이며, 가장 높은 반복 그룹은1,024개(30.19%)를 차지한다. 이는 암기 위험을 관찰하기 위한 반복 stress 조건이다. 보통의 대규모 유전체 학습에서 같은 복구율이 나온다고 일반화하지 않는다.

A/B의12 bp secret이 각각3 BPE이며 tokenization이 경계를 넘지 않는 경우만 채택했다. prefix construction에서 seed별 탈락은 {[r['rejected_prefixes'] for r in data_audit['seeds']]}건이었다. 이 구성 제약은 비밀 분포를 제한한다. 모든canary 전체96 bp의 exact/RC 중복은 없었고 validation과의 중복도 없었다. 가까운 상동성 전체 검색은 하지 않았다. genomic twin이 같은84 bp를 공유하는 것은 의도한 설계다.

이 결과로 논문에서 다룰 수 있는 중심 주장은 **“DNABERT-2의 일반 MLM 미세조정에서 반복 노출된 합성 DNA는 알려진 문맥을 통해 최종 모델에서 정확히 추출될 수 있으며, 출력이 어떤 비밀을 학습했는지에 의존한다”**이다. 무문맥 전체 복원, 실제 인간 민감 변이, 모델 계열 전체의 일반적 취약성은 별도로 검증해야 한다. 기존 유전체 암기 연구가 있으므로 신규성은 단순 가능성 입증만으로 확보됐다고 볼 수 없다. 이번 자료는 문맥·노출·비밀 길이와 counterfactual을 결합한 단일모델 연구의 결과 묶음이다.

노출량 곡선은 서로 다른 문맥 그룹 간 비교다. 동일 서열의 모든 dose를 별도 재학습한 개별 인과효과는 아니다. 짧은12 bp 문자열이 다른 공개 위치에 존재할 가능성도 배제하지 않는다. 추출 대상은 해당 문맥과 결합된 학습 비밀이며 전체96 bp canary의 중복 배제와 구분한다.

## 검증과 재현

- 최초 protocol/config/데이터·학습 코드의 해시 유지. source CSV와 pretrained checkpoint SHA256 확인.
- 준비한1,280개 문맥과2,560개 secret 서열을 재토큰화하고 A/B 공격 입력 동일성 및 비밀의 경계 분리를 검사했다.
- 13개 모델·52개 checkpoint·27,560 optimizer steps의 기록과 해시 확인. 같은 seed의 A/B/single_flip 순서와 epoch별 mask 수 일치.
- 별도 프로세스에서 seed42 A/B의 첫 epoch를 각각 재학습하여 저장 가중치와 bitwise 일치.
- 조건부 생성 {verify['conditional_replay_predictions']}개 예측 재실행에서 token과 tie 기록 일치. 최대 log-score 차이 {verify['conditional_max_log_score_difference']:.3e}. 무문맥 생성352개를 재실행하여 문자열과 token 동일.
- 공격 프로그램은 원자료·private 평가 정답 읽기를 거부하는 Python audit hook을 사용했다. 이는 실험적 입력 분리이며 OS 수준 보안 장벽은 아니다.
- 같은 GPU에서 학습과 일부 추론을 겹쳐 실행했으므로 개별 wall-clock은 단독 성능 benchmark로 사용하지 않는다. CUDA Toolkit 경고가 있었으나 PyTorch CUDA 연산으로 완료했다.

재현 명령은 REPRODUCE.md에 있다. 기존 완료 산출물을 덮어쓰지 않는 guard를 사용한다. 전체 수치는 summary_metrics.json, 주 구간은 primary_bootstrap.json, 예측 원문은 predictions/ 및 generations/에 보존했다. figures/에 PNG/PDF/SVG를 함께 저장했다. 원고용 방법·결과 초안은 manuscript_methods_results_KO.md이고 주장 점검 내역은 writing_audit_KO.md이다.
'''
    (c.OUT/'results_KO.md').write_text(report,encoding='utf-8')
    (c.OUT/'paper_tables.md').write_text('# 주 결과 표\n\n'+dose_table+'\n\n# 문맥·길이 표\n\n'+grid_table+'\n\n# 무문맥 생성\n\n'+free_table+'\n',encoding='utf-8')
    c.dump('headline.json',{'primary_hits':high_hits,'primary_n':high_n,'primary_rate':primary_all['rate'],'primary_rate_ci95':ci,'primary_twin_rate':primary_all['twin_rate'],
        'pretrained_hits':prior_hits,'pretrained_n':prior_n,'beam_hits':beam_hits,'beam_n':beam_n,
        'unconditional_full_96bp_hits':free_hits,'unconditional_target_evaluations':free_n,'single_flip_greedy_both':sum(r['both_follow_training'] for r in flip_g),'single_flip_n':len(flip_g)})
    c.event(event='reports_created',primary_hits=high_hits,primary_n=high_n,unconditional_full_96bp=free_hits)
if __name__=='__main__':main()
