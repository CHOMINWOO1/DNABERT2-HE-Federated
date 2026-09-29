from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=ROOT / "experiments/privacy_pilot_20260923/run_v1")
    args = parser.parse_args()
    root = args.run
    cfg = read(root / "run_config.json")
    assert (root / "DONE.json").exists(), "Main experiment is not complete"
    records, audits = [], []
    for seed in cfg["seeds"]:
        folder = root / f"seed_{seed}"
        data = read(folder / "data_manifest.json")
        challenges = read(folder / "attacker_challenges.json")
        truth = read(folder / "evaluator_truth.json")
        assert len(challenges) == 8 * cfg["per_group"]
        assert all(set(c) == {"id", "prefix_ids", "hidden_token_count"} for c in challenges)
        assert len({c["sequence"] for c in data["canaries"]}) == len(data["canaries"])
        heldout = {c["sequence"] for c in data["canaries"] if c["repeat_per_epoch"] == 0}
        train = {c["sequence"] for c in data["canaries"] if c["repeat_per_epoch"] > 0}
        train.update(p["sequence"] for p in data["public"])
        assert not heldout.intersection(train)
        for mode in ["pretrained"] + cfg["modes"]:
            arm = folder / mode
            assert (arm / "DONE.json").exists()
            epoch = 0 if mode == "pretrained" else cfg["epochs"]
            preds = read(arm / f"predictions_epoch_{epoch:03d}.json")
            assert len(preds) == 2 * len(challenges)
            assert all(p["exact"] == int(p["true_secret"] == p["predicted_secret"]) for p in preds)
            assert all(p["true_secret"] == truth[p["challenge_id"]]["secret_dna"] for p in preds)
            assert all(p["hidden_bp"] + p["known_bp"] == 96 for p in preds)
            records.extend(preds)
        positive = folder / "positive_control"
        if (positive / "DONE.json").exists():
            preds = read(positive / f"predictions_epoch_{cfg['positive_steps']:03d}.json")
            assert all(p["exact"] == int(p["true_secret"] == p["predicted_secret"]) for p in preds)
            records.extend(preds)
        audits.append({"seed": seed, "heldout_absent_from_training": True,
                       "attacker_artifact_has_no_ground_truth": True,
                       "prediction_scores_recomputed": True})
    groups = collections.defaultdict(list)
    for r in records:
        groups[(r["mode"],r["repeat_per_epoch"],r["hidden_tokens"],r["attack"])].append(r)
    summary = []
    for (mode, repeat, k, attack), group in sorted(groups.items()):
        summary.append({"mode":mode,"repeat_per_epoch":repeat,"hidden_tokens":k,"attack":attack,
                        "n":len(group),"recovered":sum(r["exact"] for r in group),
                        "exact_rate":float(np.mean([r["exact"] for r in group])),
                        "mean_hidden_bp":float(np.mean([r["hidden_bp"] for r in group])),
                        "min_hidden_bp":min(r["hidden_bp"] for r in group),
                        "max_hidden_bp":max(r["hidden_bp"] for r in group),
                        "edit_similarity":float(np.mean([r["edit_similarity"] for r in group])),
                        "per_seed":{str(s):{"n":sum(r["seed"]==s for r in group),
                            "recovered":sum(r["exact"] for r in group if r["seed"]==s)} for s in sorted({r["seed"] for r in group})}})
    (root / "aggregate_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    fields = [k for k in summary[0] if k != "per_seed"]
    with (root / "aggregate_summary.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer = csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary)
    (root / "data_audit.json").write_text(json.dumps(audits,indent=2),encoding="utf-8")

    colors = {"pretrained":"#75859A","fullft":"#176B87","lora8":"#DD8452"}
    names = {"pretrained":"Pretrained","fullft":"Full fine-tuning","lora8":"LoRA rank 8"}
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig, axes = plt.subplots(1,2,figsize=(12,4.7),sharey=True)
    repeat_values = [0,1,4,16]
    for ax,k in zip(axes,[1,3]):
        for j,mode in enumerate(["pretrained","fullft","lora8"]):
            rows = [next(r for r in summary if r["mode"]==mode and r["repeat_per_epoch"]==rep
                         and r["hidden_tokens"]==k and r["attack"]=="greedy") for rep in repeat_values]
            xx = np.arange(4) + (j-1)*0.25
            yy = [r["exact_rate"]*100 for r in rows]
            ax.bar(xx,yy,width=0.23,label=names[mode],color=colors[mode],alpha=0.88)
            for pos,r in zip(xx,rows):
                label_y = 103 if r["exact_rate"] > .9 else r["exact_rate"]*100
                ax.annotate(f"{r['recovered']}/{r['n']}",(pos,label_y),
                            xytext=(0,5),textcoords="offset points",ha="center",fontsize=8)
                for offset,(s,v) in zip([-.05,0,.05],r["per_seed"].items()):
                    ax.scatter(pos+offset,100*v["recovered"]/v["n"],s=12,color="#202C39",zorder=4)
        lengths = [r["hidden_bp"] for r in records if r["mode"]=="pretrained" and r["hidden_tokens"]==k and r["attack"]=="greedy"]
        ax.set_title(f"{k} hidden BPE token{'s' if k>1 else ''} ({min(lengths)}–{max(lengths)} bp)")
        ax.set_xticks(range(4),["Unseen", "20", "80", "320"])
        ax.set_xlabel("Total training presentations per canary")
        ax.set_ylim(0,110)
        ax.grid(axis="y",alpha=.15)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Exact hidden-suffix recovery (%)")
    axes[0].legend(frameon=False,loc="upper left")
    fig.suptitle("DNABERT-2 final-model extraction: known prefix + hidden token count",fontsize=13)
    fig.text(.5,.01,"Greedy decoding; n=72 per group across 3 seeds. Dots: each seed. Ordinary dynamic MLM, 20 epochs.",ha="center",fontsize=9)
    fig.tight_layout(rect=[0,.04,1,.94])
    fig.savefig(root / "recovery_main.png",dpi=180)
    fig.savefig(root / "recovery_main.pdf")
    plt.close(fig)

    # Primary endpoint table, explicit diagnostic control and concrete examples.
    lines = ["# 유전체 모델 최종 가중치 복구 파일럿 결과", "", "실행일: 2026-09-23. 탐색적 파일럿이며 논문 신규성 또는 일반적 안전성의 증명이 아니다.", "",
             "## 주 실험: 일반 MLM 미세조정", "",
             "DNABERT-2의 사전학습 가중치, FullFT, LoRA r8을 비교했다. 3개 seed(42/43/44), 그룹별 총 72개 합성 서열이다. 공개 DNA와 96 bp 합성 canary를 함께 학습했다. 공격자는 정확한 prefix와 숨긴 BPE 토큰 개수를 알고, 모델만 사용해 숨긴 접미사를 생성한다. 정답을 입력하거나 정답 후보 목록을 제공하지 않았다.", "",
             "| 숨긴 토큰 | 총 노출 횟수 | 사전학습 모델 | FullFT | LoRA r8 |", "|---:|---:|---:|---:|---:|"]
    for k in [1,3]:
        for rep in repeat_values:
            row = []
            for mode in ["pretrained","fullft","lora8"]:
                r = next(r for r in summary if (r["mode"],r["repeat_per_epoch"],r["hidden_tokens"],r["attack"])==(mode,rep,k,"greedy"))
                row.append(f"{r['recovered']}/{r['n']} ({100*r['exact_rate']:.1f}%)")
            lines.append(f"| {k} | {rep*cfg['epochs'] if rep else '미노출'} | " + " | ".join(row) + " |")
    lines += ["", "단위는 숨긴 접미사 전체의 정확 일치이다. 공개 prefix는 성과에 포함하지 않았다. 1-token 조건은 3-token 조건보다 더 많은 정답 문맥을 제공하므로 서로 같은 난도의 문제가 아니다.", ""]
    for k in [1,3]:
        hidden = [r["hidden_bp"] for r in records if r["mode"]=="pretrained" and r["hidden_tokens"]==k and r["attack"]=="greedy"]
        lines.append(f"- {k}-token 조건: 숨긴 길이 {min(hidden)}–{max(hidden)} bp, 평균 {np.mean(hidden):.2f} bp. 나머지 {96-max(hidden)}–{96-min(hidden)} bp는 공격자가 이미 안다.")
    lines += ["", "![정확 복구율](recovery_main.png)", "", "## 의도적 과적합 양성 대조", "",
              "Seed 42의 학습 canary 72개만 사용해 접미사 3-token을 항상 가리고 100 steps 학습했다. 이는 공격 경로의 작동 여부를 확인하는 별도 진단이며 일반 MLM 결과에 합치지 않는다. 미노출 canary 24개는 여기서도 사용하지 않았다.", "",
              "| 공격 | 대상 | 정확 복구 |", "|---|---|---:|"]
    for attack in ["greedy","parallel"]:
        for exposed in [True,False]:
            subset = [r for r in records if r["mode"]=="positive_control" and r["hidden_tokens"]==3 and r["attack"]==attack and (r["repeat_per_epoch"]>0)==exposed]
            n, count = len(subset),sum(r["exact"] for r in subset)
            if n:
                lines.append(f"| {attack} | {'학습' if exposed else '미학습'} | {count}/{n} ({100*count/n:.1f}%) |")
    lines += ["", "## 복구 예시", "", "표에는 숨긴 접미사만 표시한다. 모두 생성한 합성 DNA이고 환자 정보가 아니다.", "",
              "| 모델/조건 | 표본 | 숨긴 정답 | 모델 출력 | 일치 |", "|---|---|---|---|---|"]
    examples = []
    for mode, exact in [("fullft",1),("lora8",1),("fullft",0),("positive_control",1)]:
        pool = [r for r in records if r["mode"]==mode and r["hidden_tokens"]==3 and r["attack"]=="greedy" and r["repeat_per_epoch"]>0 and r["exact"]==exact]
        examples.extend(pool[:2])
    for r in examples:
        lines.append(f"| {r['mode']} | {r['challenge_id']} | `{r['true_secret']}` | `{r['predicted_secret']}` | {'O' if r['exact'] else 'X'} |")
    lines += ["", "## 해석 범위", "",
              "- 최종 모델에서 알려진 문맥을 이용해 짧은 합성 접미사를 복구한 결과다. 전체 유전체의 무단 복원, 환자 신원, 질병·SNP 복구를 입증하지 않는다.",
              "- 동일 분포의 미노출 canary와 사전학습 모델을 대조했지만 동일 비밀의 포함/제외 재학습은 아직 없다. 그룹별 secret 길이·조성 차이도 후속 통제 대상이다.",
              "- FullFT와 LoRA는 학습률·학습 용량·head 학습 여부가 다르다. 비교 결과를 rank만의 인과효과 또는 LoRA의 안전성으로 일반화할 수 없다.",
              "- 공개 DNA 표본이 작고 20 epochs, 최대 320회 노출인 스트레스 조건이다. 현실적인 대규모 임상 학습의 유출률로 해석하지 않는다.",
              "- 중간 5-epoch 결과도 저장했지만 주 표는 사전에 고정한 20-epoch 결과다. 공격은 greedy가 주 분석이며 parallel은 보조이다.",
              "- CKKS·연합 집계·gradient inversion을 실행하지 않았다. 이 결과는 HE의 방어 효과 평가가 아니다.", "",
              "## 재현", "", "```powershell", "& .tmp/privacy_runtime/python311/python.exe -u scripts/run_privacy_canary_pilot.py", "& .tmp/privacy_runtime/python311/python.exe scripts/analyze_privacy_canary_pilot.py", "```", "",
              "프로토콜은 상위 protocol_KO.md, 문헌 범위는 literature_scope_KO.md에 있다. 각 seed 폴더의 data_manifest, attacker_challenges, evaluator_truth, predictions, training_log, final_trainable_state, DONE.json으로 데이터·예측·가중치를 추적한다. 원본과 코드 SHA256을 저장했다."]
    counter_file = root.parent / "counterfactual_v1/paired_summary.json"
    if counter_file.exists():
        paired = read(counter_file)
        lines += ["", "## 사후 탐색: 동일 비밀의 포함·제외 교환", "",
                  "최초 결과를 본 뒤 protocol을 별도로 기록하고 추가했다. 같은 합성 서열·공개 DNA·공격 입력을 유지하고 0회와 16회/epoch 노출 그룹을 서로 교환해 FullFT를 새로 학습했다. 아래는 같은 비밀을 두 모델에서 비교한 결과이다. 총 학습량은 같지만 두 그룹을 함께 바꿨으므로 단일 표본 leave-one-out은 아니다.", "",
                  "| 같은 서열의 학습 노출 변경 | 변경 전 정확 복구 | 변경 후 정확 복구 |", "|---|---:|---:|"]
        for row in paired:
            if row["attack"] == "greedy" and row["hidden_tokens"] == 3:
                lines.append(f"| {row['old_repeat']*20} → {row['new_repeat']*20}회 | {row['original_recovered']}/{row['n']} | {row['swapped_recovered']}/{row['n']} |")
        lines += ["", "이 교환은 복구가 학습 노출에 의존한다는 추가 근거다. 합성 접미사의 조건부 복구이며 개인 유전체·변이 복구의 증거는 아니다. 원자료는 ../counterfactual_v1/paired_comparisons.json에 있다."]
        paired_rows = read(counter_file.parent / "paired_comparisons.json")
        lines += ["", "교환 후 새로 학습에 포함한 그룹(0→320회)의 3-token 복구는 seed별로 다음과 같다.", ""]
        for seed in cfg["seeds"]:
            subset = [r for r in paired_rows if r["seed"]==seed and r["hidden_tokens"]==3 and r["attack"]=="greedy" and r["old_repeat"]==0]
            log = read(counter_file.parent / f"seed_{seed}/fullft/training_log.json")
            lines.append(f"- Seed {seed}: {sum(r['swapped_exact'] for r in subset)}/{len(subset)} 복구, 마지막 epoch MLM 학습 loss {log[-1]['masked_token_loss']:.3f}.")
        lines += ["", "Seed 44 교환 실험은 학습 loss가 중간에 다시 높아져 마지막 6.680에 머물렀고 새 노출 그룹 복구도 0/24였다. 이 실행을 제외하거나 성공한 재실행으로 대체하지 않았다. 학습 안정성·적합도에 따라 복구 결과가 달라지므로 노출 횟수만으로 모든 경우의 복구율을 예측할 수 없다."]
        lines = [line.replace("동일 분포의 미노출 canary와 사전학습 모델을 대조했지만 동일 비밀의 포함/제외 재학습은 아직 없다. 그룹별 secret 길이·조성 차이도 후속 통제 대상이다.",
                              "동일 분포의 미노출 canary와 사전학습 모델을 대조하고, 사후 탐색으로 동일 비밀의 그룹 단위 포함/제외 교환도 수행했다. 엄밀한 단일 표본 삭제 대조와 개인 변이 검증은 후속 과제다.") for line in lines]
    replay = root / "replay_DONE.json"
    if replay.exists():
        check = read(replay)
        lines += ["", "## 최종 가중치 재실행 검증", "",
                  f"별도 Python 프로세스에서 저장된 가중치를 새 모델에 불러와 {check['arms']}개 arm, {check['prediction_cases']:,}개 예측을 다시 생성했다. 공격 입력에 정답을 전달하지 않았고, 생성 후 원결과와 비교해 토큰 단위로 전부 일치했다. 체크포인트 SHA256과 학습 가능 파라미터의 저장 범위도 확인했다. replay_audit.json에 검증 결과가 있다."]
    lines += ["", "## 독립적인 사전연구로 발전시키기 위한 판단", "",
              "이번 결과는 강한 반복 노출의 짧은 합성 DNA에서 조건부 정확 복구가 가능한 실행 사례다. 유전체 모델의 canary 추출과 LoRA 암기는 이미 선행연구가 있으므로, 이 표만으로 신규 취약성을 발견했다고 주장할 수 없다.", "",
              "1. 학습 적합도를 맞춘 비교가 우선이다. 현재 LoRA는 본 학습 loss도 FullFT보다 높다. LR·학습량·rank·MLM head 학습 여부를 분리하고, 유사한 일반화 성능에서 유출 차이가 유지되는지 확인해야 한다.",
              "2. 실제 민감 정보의 대리 지표를 강화해야 한다. 동일 공개 문맥에 숨긴 합성 SNP/indel을 무작위 배정하고, 참조서열·대립유전자 빈도·사전학습 모델만으로 얻는 성능을 넘는지 평가하는 연구 질문을 검토할 수 있다.",
              "3. 공격자의 문맥 지식과 생성 예산을 바꿔야 한다. 현재 96 bp 중 대부분을 알고 토큰 개수까지 받는다. 더 적은 문맥, 더 긴 비밀, 비밀 길이를 모르는 조건 및 여러 생성 공격을 평가해야 일반적 위험에 가까워진다.",
              "4. 두 논문의 연결은 위협모델을 맞춰야 한다. 현재 실험은 최종 모델의 기억에 의한 유출이다. CKKS 집계는 학습 중 개별 업데이트의 기밀성을 다루므로 이 최종 모델 유출의 직접 방어가 아니다. HE 후속 논문으로 연결하려면 첫 연구에 개별 업데이트·집계 후 업데이트의 실제 노출 위험을 별도 포함하거나, 후속 연구에서 최종 모델 유출을 다루는 별도 보호 기법을 추가해야 한다.", "",
              "따라서 판정은 '후속 연구 설계를 뒷받침하는 유의미한 파일럿 결과 확보'이며, '독립 논문에 충분한 신규성·현실성 검증 완료'는 아니다."]
    losses = []
    for seed in cfg['seeds']:
        for mode in cfg['modes']:
            trace = read(root / f'seed_{seed}/{mode}/training_log.json')
            losses.append({'seed':seed,'mode':mode,'last_train_mlm_loss':trace[-1]['masked_token_loss']})
    (root / 'final_training_losses.json').write_text(json.dumps(losses,indent=2),encoding='utf-8')
    lines += ['', '동일 20-epoch 예산의 마지막 MLM 학습 loss:', '', '| Seed | FullFT | LoRA r8 |', '|---:|---:|---:|']
    for seed in cfg['seeds']:
        vals = {r['mode']:r['last_train_mlm_loss'] for r in losses if r['seed']==seed}
        lines.append(f"| {seed} | {vals['fullft']:.3f} | {vals['lora8']:.3f} |")
    lines += ['', '이 수치는 마스킹한 학습 토큰에서 측정한 loss이며 독립 test utility가 아니다. LoRA의 복구율이 낮다는 사실을 동일 성능의 프라이버시 우위로 해석할 근거는 아직 없다.']
    (root / "results_KO.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"summary":str(root / "results_KO.md"),"main_k3_greedy": [{k:s[k] for k in ('mode','repeat_per_epoch','n','recovered')} for s in summary if s["attack"]=="greedy" and s['hidden_tokens']==3]},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
