import sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import core as c
required=['training_DONE.json','conditional_attack_DONE.json','generation_DONE.json','verification.json','results_KO.md','manuscript_methods_results_KO.md','headline.json','visual_review.json']
for name in required:assert (c.OUT/name).exists(),name
manifest={}
for p in sorted(c.OUT.rglob('*')):
    if not p.is_file() or any(x in ('tmp','__pycache__') for x in p.relative_to(c.OUT).parts):continue
    if p.parent==c.OUT and p.name in ('DONE.json','completion_manifest.json'):continue
    manifest[str(p.relative_to(c.OUT))]=c.sha(p)
c.dump('completion_manifest.json',{'unix':time.time(),'artifacts':manifest})
c.dump('DONE.json',{'status':'complete','unix':time.time(),'model_runs':13,'seeds':5,'canary_contexts':1280,'canary_sequences':2560,
    'artifact_count':len(manifest),'manifest_sha256':c.sha(c.OUT/'completion_manifest.json'),'scope':'single-backbone controlled final-weight memorization and extraction study; no FL/HE'})
c.event(event='study_complete',artifacts=len(manifest))
