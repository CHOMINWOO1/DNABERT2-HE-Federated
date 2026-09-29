"""Recompute predictions from saved final weights in a fresh process.

Read ground-truth-containing result files only AFTER attack prediction completes.
"""
from pathlib import Path
import argparse
import gc
import json
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / "scripts"))
import run_privacy_canary_pilot as p


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run",type=Path,default=ROOT / "experiments/privacy_pilot_20260923/run_v1")
    args=parser.parse_args()
    cfg=json.loads((args.run / "run_config.json").read_text(encoding="utf-8"))
    assert (args.run / "DONE.json").exists()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True
    tokenizer=p.AutoTokenizer.from_pretrained(p.SNAPSHOT,local_files_only=True,trust_remote_code=True)
    dna_ids=torch.tensor(sorted(i for token,i in tokenizer.get_vocab().items() if token and set(token)<=set("ACGT")))
    verified=[]
    for seed in cfg["seeds"]:
        folder=args.run / f"seed_{seed}"
        challenges=json.loads((folder / "attacker_challenges.json").read_text(encoding="utf-8"))
        assert all(set(c)=={"id","prefix_ids","hidden_token_count"} for c in challenges)
        modes=cfg["modes"] + (["positive_control"] if (folder / "positive_control/DONE.json").exists() else [])
        for mode in modes:
            arm=folder / mode
            saved=json.loads((arm / "DONE.json").read_text(encoding="utf-8"))
            checkpoint=arm / "final_trainable_state.pt"
            assert p.sha(checkpoint)==saved["checkpoint_sha256"]
            model,_=p.load_model("fullft" if mode=="positive_control" else mode,seed)
            state=torch.load(checkpoint,map_location="cpu",weights_only=True)
            wanted={n for n,t in model.named_parameters() if t.requires_grad}
            assert set(state)==wanted, f"Checkpoint does not cover trainable parameters: {mode}"
            result=model.load_state_dict(state,strict=False)
            assert not result.unexpected_keys
            assert not (set(result.missing_keys)&wanted)
            # Reconstruct from a final model and public challenges only.
            predicted=p.extract_suffixes(model,challenges,tokenizer,dna_ids)
            epoch=cfg["positive_steps"] if mode=="positive_control" else cfg["epochs"]
            expected=json.loads((arm / f"predictions_epoch_{epoch:03d}.json").read_text(encoding="utf-8"))
            mismatches=[r["challenge_id"]+":"+r["attack"] for r in expected
                        if predicted[r["challenge_id"]][r["attack"]]!=r["predicted_ids"]]
            item={"seed":seed,"mode":mode,"prediction_cases":len(expected),"mismatches":mismatches,
                  "checkpoint_hash_verified":True,"fresh_model_reload":True,
                  "ground_truth_read_only_after_prediction":True}
            verified.append(item)
            p.dump(args.run / "replay_audit.json",verified)
            print(json.dumps(item),flush=True)
            assert not mismatches, "Saved-checkpoint replay differs from recorded outputs"
            del state,model
            gc.collect()
            torch.cuda.empty_cache()
    p.dump(args.run / "replay_DONE.json",{"passed":True,"arms":len(verified),
             "prediction_cases":sum(x["prediction_cases"] for x in verified),
             "verifier_sha256":p.sha(Path(__file__))})


if __name__=="__main__":
    main()
