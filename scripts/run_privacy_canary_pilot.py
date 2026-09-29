"""Controlled final-model DNA suffix extraction pilot; no patient data.

Attack routines receive public prefix token IDs and hidden token count only.
Ground truth is used only by training and by the separate scoring routine.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers.models.bert.configuration_bert import BertConfig

SNAPSHOT = ROOT / ".cache/huggingface/hub/models--zhihan1996--DNABERT-2-117M/snapshots/7bce263b15377fc15361f52cfab88f8b586abda0"


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dna_decode(tokenizer, ids):
    return "".join(tokenizer.convert_ids_to_tokens([int(x) for x in ids]))


def load_model(mode, seed):
    seed_all(seed)
    config = BertConfig.from_pretrained(SNAPSHOT, local_files_only=True)
    # Cached implementation uses this nonzero setting to avoid its old Triton kernel.
    config.attention_probs_dropout_prob = 0.1
    model, load_info = AutoModelForMaskedLM.from_pretrained(
        SNAPSHOT, config=config, trust_remote_code=True, local_files_only=True,
        output_loading_info=True,
    )
    if load_info["missing_keys"] or load_info["mismatched_keys"]:
        raise RuntimeError(f"Uninitialized pretrained weights: {load_info}")
    if mode == "lora8":
        model = get_peft_model(model, LoraConfig(
            r=8, lora_alpha=16, target_modules=["Wqkv"], lora_dropout=0.05,
            bias="none",
        ))
    model = model.cuda()
    return model, load_info


def make_dataset(tokenizer, seed, per_group, public_n):
    rng = np.random.default_rng(seed)
    source = ROOT / "GUE_v2/EMP/H3K4me3/train.csv"
    with source.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    chosen = rng.choice(len(rows), public_n, replace=False)
    public = []
    for idx in chosen:
        sequence = rows[int(idx)]["sequence"].upper()
        start = (len(sequence) - 96) // 2
        crop = sequence[start:start + 96]
        assert len(crop) == 96 and set(crop) <= set("ACGT")
        public.append({"source_row": int(idx), "sequence": crop,
                       "ids": tokenizer.encode(crop, add_special_tokens=False)})
    groups = np.repeat([0, 1, 4, 16], per_group)
    rng.shuffle(groups)
    canaries = []
    used_prefixes = set()
    for i, repeat in enumerate(groups):
        while True:
            sequence = "".join(rng.choice(list("ACGT"), 96))
            ids = tokenizer.encode(sequence, add_special_tokens=False)
            assert dna_decode(tokenizer, ids) == sequence
            prefix = dna_decode(tokenizer, ids[:-3])
            if len(ids) > 8 and prefix not in used_prefixes:
                break
        used_prefixes.add(prefix)
        canaries.append({"id": f"s{seed}_c{i:03d}", "repeat_per_epoch": int(repeat),
                         "sequence": sequence, "ids": ids})
    all_public = [r["sequence"] for r in public]
    assert len({c["sequence"] for c in canaries}) == len(canaries)
    assert all(c["sequence"] not in all_public for c in canaries)
    train = [p["ids"] for p in public]
    for c in canaries:
        train.extend([c["ids"]] * c["repeat_per_epoch"])
    heldout_ids = {tuple(c["ids"]) for c in canaries if c["repeat_per_epoch"] == 0}
    assert not any(tuple(ids) in heldout_ids for ids in train)
    challenges = []
    truth = {}
    for c in canaries:
        for k in (1, 3):
            challenge_id = f"{c['id']}_k{k}"
            # This artifact deliberately contains no secret or full source sequence.
            challenges.append({"id": challenge_id, "prefix_ids": c["ids"][:-k],
                               "hidden_token_count": k})
            truth[challenge_id] = {"canary_id": c["id"], "repeat_per_epoch": c["repeat_per_epoch"],
                                   "secret_ids": c["ids"][-k:],
                                   "secret_dna": dna_decode(tokenizer, c["ids"][-k:]),
                                   "known_bp": len(dna_decode(tokenizer, c["ids"][:-k]))}
    return {"source_path": str(source), "source_sha256": sha(source), "public": public,
            "canaries": canaries, "train": train, "challenges": challenges, "truth": truth}


def pad_sequences(sequences, tokenizer):
    full = [[tokenizer.cls_token_id] + list(x) + [tokenizer.sep_token_id] for x in sequences]
    ids = torch.full((len(full), max(map(len, full))), tokenizer.pad_token_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for row, seq in enumerate(full):
        ids[row, :len(seq)] = torch.tensor(seq)
        mask[row, :len(seq)] = 1
    return ids, mask


def make_mlm_batch(sequences, tokenizer, generator, dna_ids, suffix_only=False):
    ids, attention = pad_sequences(sequences, tokenizer)
    eligible = attention.bool() & ids.ne(tokenizer.cls_token_id) & ids.ne(tokenizer.sep_token_id)
    if suffix_only:
        selected = torch.zeros_like(eligible)
        for i, seq in enumerate(sequences):
            selected[i, len(seq)-2:len(seq)+1] = True
    else:
        selected = (torch.rand(ids.shape, generator=generator) < 0.15) & eligible
        # Avoid undefined loss for a batch with no targets, without changing ordinary masks.
        if not selected.any():
            selected[0, 1] = True
    labels = ids.clone()
    labels[~selected] = -100
    if suffix_only:
        ids[selected] = tokenizer.mask_token_id
    else:
        draw = torch.rand(ids.shape, generator=generator)
        ids[selected & (draw < 0.8)] = tokenizer.mask_token_id
        replacement = selected & (draw >= 0.8) & (draw < 0.9)
        random_tokens = dna_ids[torch.randint(len(dna_ids), ids.shape, generator=generator)]
        ids[replacement] = random_tokens[replacement]
    return ids.cuda(), attention.cuda(), labels.cuda(), int(selected.sum())


@torch.inference_mode()
def extract_suffixes(model, challenges, tokenizer, dna_ids, batch_size=24):
    """No labels, secret IDs, true suffix length in bp, or candidate list received."""
    model.eval()
    allowed = dna_ids.cuda()
    predictions = {}
    for k in (1, 3):
        subset = [c for c in challenges if c["hidden_token_count"] == k]
        for start in range(0, len(subset), batch_size):
            batch = subset[start:start+batch_size]
            seqs = [c["prefix_ids"] + [tokenizer.mask_token_id] * k for c in batch]
            ids, attention = pad_sequences(seqs, tokenizer)
            ids, attention = ids.cuda(), attention.cuda()
            positions = torch.tensor([1 + len(c["prefix_ids"]) for c in batch], device="cuda")
            row = torch.arange(len(batch), device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=ids, attention_mask=attention).logits
            parallel = []
            for j in range(k):
                parallel.append(allowed[logits[row, positions+j][:, allowed].argmax(-1)])
            parallel_ids = torch.stack(parallel, 1).cpu().tolist()
            greedy = []
            current = ids.clone()
            for j in range(k):
                if j == 0:
                    current_logits = logits
                else:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        current_logits = model(input_ids=current, attention_mask=attention).logits
                predicted = allowed[current_logits[row, positions+j][:, allowed].argmax(-1)]
                greedy.append(predicted)
                current[row, positions+j] = predicted
            greedy_ids = torch.stack(greedy, 1).cpu().tolist()
            for c, p, g in zip(batch, parallel_ids, greedy_ids):
                predictions[c["id"]] = {"parallel": p, "greedy": g}
    return predictions


def edit_distance(a, b):
    row = list(range(len(b)+1))
    for i, x in enumerate(a, 1):
        nxt = [i]
        for j, y in enumerate(b, 1):
            nxt.append(min(nxt[-1]+1, row[j]+1, row[j-1]+(x != y)))
        row = nxt
    return row[-1]


def score_predictions(predictions, challenges, truth, tokenizer, mode, epoch, seed):
    records = []
    for c in challenges:
        target = truth[c["id"]]
        for attack in ("parallel", "greedy"):
            ids = predictions[c["id"]][attack]
            decoded = dna_decode(tokenizer, ids)
            expected = target["secret_dna"]
            records.append({"seed": seed, "mode": mode, "epoch": epoch, "challenge_id": c["id"],
                            "repeat_per_epoch": target["repeat_per_epoch"], "hidden_tokens": c["hidden_token_count"],
                            "attack": attack, "known_bp": target["known_bp"], "hidden_bp": len(expected),
                            "true_secret": expected, "predicted_secret": decoded,
                            "exact": int(decoded == expected),
                            "token_accuracy": sum(x == y for x, y in zip(ids, target["secret_ids"])) / len(ids),
                            "edit_similarity": 1-edit_distance(expected, decoded)/max(len(expected), len(decoded)),
                            "predicted_ids": ids})
    return records


def summarize(records):
    result = []
    keys = ("seed", "mode", "epoch", "repeat_per_epoch", "hidden_tokens", "attack")
    groups = {}
    for r in records:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    for key, group in groups.items():
        item = dict(zip(keys, key))
        item.update(n=len(group), recovered=sum(r["exact"] for r in group),
                    exact_rate=float(np.mean([r["exact"] for r in group])),
                    token_accuracy=float(np.mean([r["token_accuracy"] for r in group])),
                    edit_similarity=float(np.mean([r["edit_similarity"] for r in group])))
        result.append(item)
    return result


def evaluate_and_save(model, data, tokenizer, dna_ids, output, mode, epoch, seed):
    predictions = extract_suffixes(model, data["challenges"], tokenizer, dna_ids)
    records = score_predictions(predictions, data["challenges"], data["truth"], tokenizer, mode, epoch, seed)
    dump(output / f"predictions_epoch_{epoch:03d}.json", records)
    summary = summarize(records)
    dump(output / f"summary_epoch_{epoch:03d}.json", summary)
    headline = [{k:r[k] for k in ("repeat_per_epoch", "hidden_tokens", "recovered", "n")} for r in summary if r["attack"] == "greedy"]
    print(json.dumps({"event":"evaluation", "seed":seed, "mode":mode, "epoch":epoch, "greedy":headline}), flush=True)


def train_arm(mode, seed, data, tokenizer, dna_ids, args, out):
    arm = out / mode
    if (arm / "DONE.json").exists():
        print(f"Skipping complete {arm}", flush=True)
        return
    arm.mkdir(parents=True, exist_ok=True)
    model, load_info = load_model(mode, seed)
    trainable = {n:p for n,p in model.named_parameters() if p.requires_grad}
    lr = 5e-5 if mode == "fullft" else 2e-4
    dump(arm / "config.json", {"seed":seed,"mode":mode,"learning_rate":lr,"epochs":args.epochs,
                              "batch_size":args.batch_size,"trainable_parameters":sum(p.numel() for p in trainable.values()),
                              "loading_info":load_info,"optimizer":"AdamW","weight_decay":0.01,
                              "objective":"standard dynamic MLM 15%; 80% mask / 10% random / 10% unchanged",
                              "lora_rank":8 if mode == "lora8" else None,
                              "masking_and_order_seed":seed+991})
    optimizer = torch.optim.AdamW(trainable.values(), lr=lr, weight_decay=0.01)
    generator = torch.Generator().manual_seed(seed + 991)
    order_rng = np.random.default_rng(seed + 991)
    data_train = data["train"]
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    logs = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = order_rng.permutation(len(data_train))
        losses, targets = [], 0
        for start in range(0, len(order), args.batch_size):
            seqs = [data_train[int(i)] for i in order[start:start+args.batch_size]]
            ids, mask, labels, count = make_mlm_batch(seqs, tokenizer, generator, dna_ids)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss: {mode}, {seed}, epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable.values(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()) * count)
            targets += count
        log = {"epoch":epoch,"masked_token_loss":sum(losses)/targets,"targets":targets,
               "elapsed_seconds":time.time()-started}
        logs.append(log)
        dump(arm / "training_log.json", logs)
        print(json.dumps({"event":"train","seed":seed,"mode":mode,**log}), flush=True)
        if epoch in {5, args.epochs}:
            evaluate_and_save(model, data, tokenizer, dna_ids, arm, mode, epoch, seed)
    weights = {n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
    torch.save(weights, arm / "final_trainable_state.pt")
    dump(arm / "DONE.json", {"elapsed_seconds":time.time()-started,"peak_gpu_bytes":torch.cuda.max_memory_allocated(),
                             "checkpoint_sha256":sha(arm / "final_trainable_state.pt"),
                             "source_sha256":sha(Path(__file__))})
    del optimizer, weights, model
    gc.collect()
    torch.cuda.empty_cache()


def positive_control(seed, data, tokenizer, dna_ids, args, out):
    """Deliberately overfit prefix -> three-token suffix; diagnostic, not ordinary training."""
    arm = out / "positive_control"
    if (arm / "DONE.json").exists():
        return
    arm.mkdir(parents=True, exist_ok=True)
    model, _ = load_model("fullft", seed)
    train = [c["ids"] for c in data["canaries"] if c["repeat_per_epoch"] > 0]
    generator = torch.Generator().manual_seed(seed + 171)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    rng = np.random.default_rng(seed+171)
    logs = []
    started = time.time()
    for step in range(1, args.positive_steps+1):
        model.train()
        indices = rng.choice(len(train), min(args.batch_size,len(train)), replace=False)
        seqs = [train[int(i)] for i in indices]
        ids, mask, labels, _ = make_mlm_batch(seqs,tokenizer,generator,dna_ids,suffix_only=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite positive-control loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        if step % 20 == 0 or step == args.positive_steps:
            logs.append({"step":step,"loss":float(loss.detach()),"elapsed_seconds":time.time()-started})
            dump(arm / "training_log.json", logs)
            print(json.dumps({"event":"positive_control","seed":seed,**logs[-1]}),flush=True)
    evaluate_and_save(model,data,tokenizer,dna_ids,arm,"positive_control",args.positive_steps,seed)
    torch.save({n:p.detach().cpu().clone() for n,p in model.named_parameters()},arm / "final_trainable_state.pt")
    dump(arm / "DONE.json",{"diagnostic_only":True,"steps":args.positive_steps,"learning_rate":1e-4,
                             "n_training_canaries":len(train),"elapsed_seconds":time.time()-started,
                             "checkpoint_sha256":sha(arm / "final_trainable_state.pt")})
    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=ROOT / "experiments/privacy_pilot_20260923/run_v1")
    parser.add_argument("--seeds",type=int,nargs="+",default=[42,43,44])
    parser.add_argument("--epochs",type=int,default=20)
    parser.add_argument("--per-group",type=int,default=24)
    parser.add_argument("--public-n",type=int,default=512)
    parser.add_argument("--batch-size",type=int,default=32)
    parser.add_argument("--positive-steps",type=int,default=100)
    parser.add_argument("--modes",nargs="+",default=["fullft","lora8"],choices=["fullft","lora8"])
    parser.add_argument("--skip-positive",action="store_true")
    parser.add_argument("--smoke",action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.output = args.output / "smoke"
        args.seeds, args.epochs, args.per_group, args.public_n = [987],1,2,16
        args.positive_steps = 2
    assert torch.cuda.is_available(), "CUDA required; do not silently launch long CPU training"
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    args.output.mkdir(parents=True,exist_ok=True)
    run_config = {**vars(args),"output":str(args.output),"python":sys.version,"torch":torch.__version__,
                  "gpu":torch.cuda.get_device_name(0),"snapshot":str(SNAPSHOT),"source_sha256":sha(Path(__file__))}
    config_path = args.output / "run_config.json"
    if config_path.exists():
        old = json.loads(config_path.read_text(encoding="utf-8"))
        for key in ("epochs","per_group","public_n","batch_size","positive_steps"):
            if old[key] != run_config[key]:
                raise RuntimeError(f"Refusing to overwrite run with changed {key}")
    else:
        dump(config_path,run_config)
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT,trust_remote_code=True,local_files_only=True)
    dna_ids = torch.tensor(sorted(i for token,i in tokenizer.get_vocab().items() if token and set(token) <= set("ACGT")))
    for seed in args.seeds:
        out = args.output / f"seed_{seed}"
        out.mkdir(parents=True,exist_ok=True)
        data = make_dataset(tokenizer,seed,args.per_group,args.public_n)
        dump(out / "data_manifest.json", {k:v for k,v in data.items() if k not in {"train","challenges","truth"}})
        dump(out / "attacker_challenges.json",data["challenges"])
        dump(out / "evaluator_truth.json",data["truth"])
        # Assert attacker artifact cannot directly disclose labels or secrets.
        assert all(set(c) == {"id","prefix_ids","hidden_token_count"} for c in data["challenges"])
        baseline = out / "pretrained"
        if not (baseline / "DONE.json").exists():
            model, load_info = load_model("fullft",seed)
            evaluate_and_save(model,data,tokenizer,dna_ids,baseline,"pretrained",0,seed)
            dump(baseline / "DONE.json",{"loading_info":load_info})
            del model
            gc.collect()
            torch.cuda.empty_cache()
        for mode in args.modes:
            train_arm(mode,seed,data,tokenizer,dna_ids,args,out)
        if not args.skip_positive and seed == args.seeds[0]:
            positive_control(seed,data,tokenizer,dna_ids,args,out)
    dump(args.output / "DONE.json",{"seeds":args.seeds,"completed_at_unix":time.time(),"source_sha256":sha(Path(__file__))})


if __name__ == "__main__":
    main()
