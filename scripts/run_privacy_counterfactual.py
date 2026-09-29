"""Follow-up to v1: swap unseen and high-exposure groups, keep everything else fixed."""
from pathlib import Path
import json
import sys
import time
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_privacy_canary_pilot as pilot


def main():
    original_main_data = pilot.make_dataset

    def swapped_dataset(tokenizer, seed, per_group, public_n):
        data = original_main_data(tokenizer, seed, per_group, public_n)
        for c in data["canaries"]:
            old = c["repeat_per_epoch"]
            c["original_repeat_per_epoch"] = old
            c["repeat_per_epoch"] = {0:16,16:0}.get(old,old)
        data["train"] = [p["ids"] for p in data["public"]]
        for c in data["canaries"]:
            data["train"].extend([c["ids"]] * c["repeat_per_epoch"])
        for t in data["truth"].values():
            old = t["repeat_per_epoch"]
            t["original_repeat_per_epoch"] = old
            t["repeat_per_epoch"] = {0:16,16:0}.get(old,old)
        return data

    pilot.make_dataset = swapped_dataset
    output = ROOT / "experiments/privacy_pilot_20260923/counterfactual_v1"
    sys.argv = [sys.argv[0], "--output",str(output),"--seeds","42","43","44","--modes","fullft","--skip-positive"]
    pilot.main()
    evidence = {"wrapper_sha256":pilot.sha(Path(__file__)),"base_script_sha256":pilot.sha(ROOT / "scripts/run_privacy_canary_pilot.py"),
                "design":"Swap 0 and 16 repeats per epoch for the identical canaries; 1/4 groups and public DNA unchanged.",
                "posthoc_followup_to":"run_v1","completed_at_unix":time.time()}
    pilot.dump(output / "counterfactual_provenance.json",evidence)
    comparisons=[]
    for seed in [42,43,44]:
        original = ROOT / f"experiments/privacy_pilot_20260923/run_v1/seed_{seed}"
        swapped = output / f"seed_{seed}"
        assert json.loads((original / "attacker_challenges.json").read_text()) == json.loads((swapped / "attacker_challenges.json").read_text())
        orig_data=json.loads((original / "data_manifest.json").read_text())
        swap_data=json.loads((swapped / "data_manifest.json").read_text())
        assert orig_data["public"] == swap_data["public"]
        assert [x["sequence"] for x in orig_data["canaries"]] == [x["sequence"] for x in swap_data["canaries"]]
        old=json.loads((original / "fullft/predictions_epoch_020.json").read_text())
        new=json.loads((swapped / "fullft/predictions_epoch_020.json").read_text())
        index={(r["challenge_id"],r["attack"]):r for r in new}
        for r in old:
            q=index[r["challenge_id"],r["attack"]]
            assert r["true_secret"]==q["true_secret"]
            comparisons.append({"seed":seed,"id":r["challenge_id"],"hidden_tokens":r["hidden_tokens"],"attack":r["attack"],
                                "old_repeat":r["repeat_per_epoch"],"new_repeat":q["repeat_per_epoch"],
                                "original_exact":r["exact"],"swapped_exact":q["exact"],
                                "secret":r["true_secret"],"original_prediction":r["predicted_secret"],
                                "swapped_prediction":q["predicted_secret"]})
    pilot.dump(output / "paired_comparisons.json",comparisons)
    summary=[]
    for attack in ["greedy","parallel"]:
        for k in [1,3]:
            for old in [0,16]:
                group=[r for r in comparisons if r["attack"]==attack and r["hidden_tokens"]==k and r["old_repeat"]==old]
                summary.append({"attack":attack,"hidden_tokens":k,"old_repeat":old,"new_repeat":16-old,"n":len(group),
                                "original_recovered":sum(r["original_exact"] for r in group),
                                "swapped_recovered":sum(r["swapped_exact"] for r in group)})
    pilot.dump(output / "paired_summary.json",summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":
    main()
