# Session 6 summary (20260924_031959)

Started 2026-09-24 03:19:59 UTC on n3shwd4y5g.

- train bundle: `features_v4_waterbirds_dinov2_train.npz`
- test bundle (replication): `features_v3_waterbirds_dinov2_test.npz`
- minimum group size: 0.01 of n
- no GPU work in this session.

## 00_selftest_cub_meta

- command: `python cub_meta.py --self-test`
- start: 03:20:01, end: 03:20:01, duration: 0 min 0 s
- exit code: **0** (ok)
- log: `results/logs/session6_20260924_031959/00_selftest_cub_meta.log`
- new files in results/: 

```
cub_meta self-test
  ok   find_cub locates a real-shaped tree
attribute labels: 100%|██████████| 36/36 [00:00<00:00, 338553.69line/s]
  ok   attributes recovered exactly; the six-field line is counted, not dropped
  ok   part visibility recovered exactly
  ok   attribute names read from the archive root
attribute labels: 100%|██████████| 35/35 [00:00<00:00, 343795.41line/s]
  ok   a missing (image, attribute) pair is caught
scan archive: 11member [00:00, 8155.80member/s]
  ok   archive extraction takes the 5 text files and no image
SELF-TEST OK
```

## 01_selftest_sv_screen

- command: `python sv_screen.py --self-test`
- start: 03:20:01, end: 03:20:04, duration: 0 min 3 s
- exit code: **0** (ok)
- log: `results/logs/session6_20260924_031959/01_selftest_sv_screen.log`
- new files in results/: 

```
sv_screen self-test
  ok   exact label-stratified p (min) 0.0491 matches Monte Carlo 0.0507
  ok   exact depletion p 0.0841 matches Monte Carlo 0.0838 (k=1, expected 3.68)
  ok   Holm step-down on a known vector
  ok   small label-pure group: AUC 0.545 passes, phi +0.265 catches it
candidates: 100%|██████████| 41/41 [00:00<00:00, 1904.20cand/s]
  ok   planted margin-free group found (ratio 15.093, Holm p 4.4e-26); 0 of 40 random decoys called (|S| = 20)
  c.npz                             : 100%|██████████| 5/5 [00:00<00:00, 19275.29C/s]
  ok   unmodified group_margins.py reports the same ratio (15.093054)
  ok   detectability floor at n=4795, |S|=421, 650 tests: m >= 102
SELF-TEST OK
```

## 02_integration

- command: `python validate_session6.py`
- start: 03:20:04, end: 03:20:12, duration: 0 min 8 s
- exit code: **0** (ok)
- log: `results/logs/session6_20260924_031959/02_integration.log`
- new files in results/: 

```
session 6 integration check
checks: 100%|██████████| 7/7 [00:08<00:00,  1.20s/check]

  cub_meta.py CLI joins a mini tree
  sv_screen.py CLI finds the planted attribute, no coin-flip one
  unmodified group_margins.py measures the confirmation bundles
  sv_screen.py --compare reports agreement
  confirmation bundle features are bit-identical to the source
  the replication run on the test split confirms the planted hit
  cub_meta.py refuses a tree with a Waterbirds image missing

INTEGRATION OK -- the real command line joins the CUB metadata, finds a planted margin-free attribute and no coin-flip one, and the unmodified group_margins.py agrees with the screen.
```

## 10_bird_fraction

- reused `results/v5_bird_fraction.csv` from session 5.

## 20_cub_meta

- command: `python cub_meta.py --fractions results/v5_bird_fraction.csv`
- start: 03:20:12, end: 03:20:13, duration: 0 min 1 s
- exit code: **1** (FAILED)
- log: `results/logs/session6_20260924_031959/20_cub_meta.log`
- new files in results/: 

```
downloading CUB-200-2011 (1.2 GB; only text files will be extracted) into /notebooks/Spurious-ICLR/../data ...
download: 0B [00:00, ?B/s]
Traceback (most recent call last):
  File "/notebooks/Spurious-ICLR/cub_meta.py", line 158, in fetch_cub
    urllib.request.urlretrieve(CUB_URL, part, reporthook=hook)
  File "/usr/lib/python3.12/urllib/request.py", line 240, in urlretrieve
    with contextlib.closing(urlopen(url, data)) as fp:
                            ^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/urllib/request.py", line 215, in urlopen
    return opener.open(url, data, timeout)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/urllib/request.py", line 521, in open
    response = meth(req, response)
               ^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/urllib/request.py", line 630, in http_response
    response = self.parent.error(
               ^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/urllib/request.py", line 559, in error
    return self._call_chain(*args)
           ^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/urllib/request.py", line 492, in _call_chain
    result = func(*args)
             ^^^^^^^^^^^
  File "/usr/lib/python3.12/urllib/request.py", line 639, in http_error_default
    raise HTTPError(req.full_url, code, msg, hdrs, fp)
urllib.error.HTTPError: HTTP Error 403: Forbidden

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/notebooks/Spurious-ICLR/cub_meta.py", line 571, in <module>
    raise SystemExit(main())
                     ^^^^^^
  File "/notebooks/Spurious-ICLR/cub_meta.py", line 544, in main
    fetch = fetch_cub(root)
            ^^^^^^^^^^^^^^^
  File "/notebooks/Spurious-ICLR/cub_meta.py", line 162, in fetch_cub
    raise RuntimeError(
RuntimeError: could not download CUB-200-2011: HTTP Error 403: Forbidden
Fetch it by hand -- one public file:
    cd /notebooks/Spurious-ICLR/../data
    curl -L -o CUB_200_2011.tgz "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"
    tar xzf CUB_200_2011.tgz CUB_200_2011/images.txt CUB_200_2011/bounding_boxes.txt CUB_200_2011/attributes CUB_200_2011/parts attributes.txt

```

## ABORTED

cub_meta.py failed its structural checks or could not fetch the archive. Read `results/v6_cub_meta.md` and the log above. Nothing was screened.

## 90_push

- committing and pushing results/ (the run exited with status 1)

