# Training and evaluation protocol

The only approved detector-development sources are the official local
`data/SoccerNetGS/train` and `data/SoccerNetGS/valid` directories. The canonical
export preserves those splits exactly and records source split, sequence, and
frame for every linked image. Training entry points fail closed unless that
manifest is complete and passes the boundary checks.

`data/eval/gsr` is **LEGACY_CONTAMINATED_TEST**. Earlier ball experiments used
material from that split, so it is retained only for reproducible historical
comparisons. It is not a pristine benchmark and must never supply training,
validation, model-selection, or threshold-tuning data.

## Final holdout

No untouched target-domain holdout currently exists locally. The two
licence-documented Wikimedia clips under `data/raw` have already influenced
development, and the root demonstration video has also been used and lacks a
complete checked provenance record. Arbitrarily moving official VALID frames
would not solve this because adjacent football frames are highly correlated.

The strongest practical final benchmark is therefore a newly acquired,
manually annotated set of broadcast-style clips from entirely separate matches,
preferably Wikimedia Commons assets carrying an explicit reusable licence such
as CC BY-SA. Acquire at least six clips across at least three matches only after
the checkpoint, code, configuration, and thresholds are frozen. Split and
deduplicate at match/clip level, seal the annotations, and reveal them for one
final evaluation only. The enforceable specification and eventual asset hashes
live in `configs/evaluation/final_holdout_policy.json`; its empty `holdout_items`
array intentionally blocks pristine final-benchmark claims until acquisition.
