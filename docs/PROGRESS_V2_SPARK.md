# Progress — v2 (SPARK)

Update at the end of every task and before ending a session. Paste verification output verbatim
(trim long logs to the relevant lines).

## Done
<!-- - [S00] YYYY-MM-DD — <one line> — verification:
```
<output>
```
-->
- [S00] 2026-10-25 — image twinloop:v2 built; TotalSegmentator 2.18.0 installs cleanly; paths/config_v2/v2split + tests — verification:
```
v1 state: branch v2, v1 modules present, v1 tests 58 passed (twinloop:dev); containers: twinloop-train Up, twinloop-server Up
before/after: nnunetv2==2.8.1, torch 2.14.0a0+b2c75dd062.nv26.09 unchanged (diff -> SAME); Python 3.12.3
splits_final.json: <class 'list'> 5 ['train', 'val'] ['ImageCAS_0001', 'ImageCAS_0004', 'ImageCAS_0006']  (matches SPEC)
import totalsegmentator, sklearn, pyarrow, joblib -> ok
tests/test_v2_basics.py: 4 passed; ruff check/format clean
role counts: Counter({'v2_train': 640, 'v2_holdout': 87, 'v2_realfail': 73})
```

## In progress
WAITING ON OWNER: (1) ./scripts/dk totalseg_set_license -l <key>  (2) create the private git remote and push (SPEC §15)
Note: docker needs `sg docker -c "./scripts/dk ..."` in this shell (docker group membership not active in the session).

## BLOCKED
<!-- - [Sxx] error: … | tried: 1) … 2) … 3) … | hypothesis: … | next independent task taken: Syy -->

## Notes for next session
<!-- Observations, sanity-check results, things the owner should look at, bug reports for CLUSTER. -->
