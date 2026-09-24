# Session 6 — does any property of the photographs avoid the margin set?

Written 23 September 2026, after session 5 closed the bird-size partition. Read
`SESSION5.md` §9 first: it holds the fact this session is built on and the
insight kept for the manuscript. **There is no GPU work.** Expect about 5 minutes
of compute plus the one-time 1.2 GB CUB-200-2011 download.

---

## 1. The fact, in three lines

- `w_hat` is fitted on `(phi, y)` only. Changing the group variable does not move it.
- So for fixed features, `gamma_min / gamma_maj > 1` **if and only if** the
  larger-margin group contains **none** of the points in the margin set `S`. `S` is
  the set of points whose margin `u_i = y_i (w_hat · x_i)/||w_hat||` is within
  1e-3 (relative) of the global minimum. That 1e-3 is `group_margins.TIE_TOL`,
  the same tolerance that calls a tie.
- On the dinov2 train bundle, `|S| = 421` of 4,795. A group whose membership has
  nothing to do with margin, of size `m`, avoids all of them with probability
  about `(1 − 421/4795)^m`. That is 1e-4 at m = 100 and 1e-19 at m = 480.

Aser's reading (23 Sept): alpha > 1 is "not impossible, just almost impossible".
So it is still worth checking whether some **pre-existing** property of the
images does pick out a margin-free set. That is this session.

## 2. What the screen does

**It does NOT build a group from the margins.** Taking the points outside the
margin and calling them a group is session 5's oracle, which is circular: it
defines the group by the answer. Every candidate here is a property of the
photograph, annotated by CUB before any model existed, and the family is fixed
in `sv_screen.py` before the run:

| family | candidates | definition |
|---|---|---|
| A. CUB attributes | 312 × 2 = 624 | `G = {is_present = 1}` and `G = {is_present = 0}`, as annotated, any certainty |
| B. proxy tails | 4 × 3 × 2 = 24 | top and bottom 5/10/25% of `bird_frac`, `bbox_area_frac`, `n_visible_parts` (of CUB's 15 parts), `frac_attr_not_visible` (share of the 312 attributes marked "not visible") |
| C. framing | 2 | `bird_in_frame` (bounding box ≥ 2 px from every edge), both sides |

Species is excluded: Waterbirds' label is a function of species, so every
species group is a label proxy.

**Eligibility.** These filters use only `y` and `G`, never the margins, so
applying them before the test does not bias it:
- at least 1% of n in `G` and in the rest;
- at least 10 of each label inside `G`;
- `|AUC − 0.5| ≤ 0.10` (session 5's rule) and `|phi| ≤ 0.10` against `y`. The
  phi bound is new. AUC alone misses a small group of one label: 100 waterbirds
  and nothing else gives AUC 0.545.

**The test, in words.** The null says `G` is a random subset of the images with
`G`'s own mix of waterbirds and landbirds, unrelated to margin. The statistic is
the smallest margin inside `G`. The p-value is the exact probability that such a
random subset has a smallest margin at least that large. It is a product of two
binomial-coefficient ratios, one per label, so no simulation is needed; the
self-test checks it against one anyway. Stratifying by label stops a group's
label mix from posing as an effect. A group that contains a margin point gets
p = 1.

**Multiple testing.** Holm over all eligible candidates, family-wise α = 0.05.
A **hit** is a candidate whose group has the larger margin and whose Holm-adjusted
p is ≤ 0.05.

**A graded number, for when nothing hits.** For every candidate, the report also
gives the count of margin points inside `G` against its label-matched
expectation, with the exact probability of seeing that few or fewer. It answers
"how close did anything get".

**Detectability floor.** This is the smallest `m` at which a margin-free group
survives Holm, using the unstratified bound `C(n−|S|, m)/C(n, m)`. It is exact
for a group with the data's label mix and approximate for a skewed one. At
|S| = 421 and ~650 tests it is about m ≥ 102 (eps ≈ 2%). A smaller margin-free
group can still be a hit if its own minimum sits far above the margin, but below
the floor luck cannot be ruled out in general.

**Confirmation with the unmodified instrument.** For every hit, and always for
the three best-ranked candidates, the screen writes a bundle with `g = 1[G]`. No
rows are dropped, and `phi` and `y` are bit-identical. `group_margins.py`,
**unmodified**, then measures those bundles, and `sv_screen.py --compare` checks
that its ratios equal the screen's. They must, because the separator is
group-blind.

**Replication.** A train hit is a lead, not a result: Holm holds false positives
at 5%, not at zero. The runner re-runs the same screen on the **test split**,
which is a disjoint set of photographs with the separator re-fitted there. It
then tests only the train hits, with Holm over their number. The test split's
background mix differs from train's. That does not matter for the question,
which is whether this property of the photograph keeps its images off the
boundary. The test bundle is located automatically (the dinov2 test bundle, v4
then v3). If none is found, or the test split is not separable, this step is
skipped or fails without affecting anything else.

## 3. What was built

| file | what it is | validated by |
|---|---|---|
| `cub_meta.py` | fetch CUB-200-2011 (text files only), parse attributes / parts / boxes, join to Waterbirds by filename, compute the proxies | `--self-test`: exact recovery, an irregular six-field line is counted and not dropped, a missing (image, attribute) pair is caught, the archive yields text files and no image |
| `sv_screen.py` | the screen, `--compare`, `--replicate` | `--self-test`: exact p (min) and p (depletion) against Monte Carlo, Holm on a known vector, the phi filter catches a small single-label group, a planted margin-free group is found and 0 of 40 random decoys are called, the unmodified `group_margins.py` reports the same ratio, the detectability floor sits where the bound crosses |
| `validate_session6.py` | **the real command line** on a miniature Waterbirds + CUB tree, as subprocesses with the runner's flags | a planted attribute is a hit and no coin-flip attribute is; `group_margins.py` agrees (`--compare` → AGREEMENT OK); confirmation bundles are bit-identical to the source; the planted hit replicates on the mini test split; `cub_meta.py` refuses a tree with one image missing |
| `run_session6.sh` | the runner | `bash -n`, plus a full dry run on the miniature data, exit 0 |

**No existing script was edited.** `group_margins.py`, `regroup.py` (only its
`write_bundle` and `check_alignment` are imported), `cub_masks.py` and
`datasets.py` are untouched.

## 4. What to run

```bash
cd /notebooks/Spurious-ICLR
git pull
nohup setsid bash run_session6.sh > results/logs/session6_nohup.out 2>&1 &
```

**The script never commits or pushes** (Aser's decision, 24 Sept 2026). When it
has finished -- or aborted -- send everything back with:

```bash
cd /notebooks/Spurious-ICLR
git add -A results
git commit -m "session 6 results"
git push
```

Everything needed for review is under `results/`: `SUMMARY.md`, every step's
full log, `environment.txt`, all the `.md`/`.json` tables, and the nohup file.
This holds even for a run that died early. Not pulled, by design: the `.npz`
files (bundles and `v6_cub_meta.npz`), which are regenerable and not needed for
review.

Options: `BUNDLE=`, `TEST_BUNDLE=`, `MIN_FRAC=0.01`, `DATA_ROOT=`, `NO_DOWNLOAD=1`. Stage 0 runs the three self-checks and aborts before touching data if
any fails. If `results/v5_bird_fraction.csv` is missing on the box, the runner
recomputes it with `cub_masks.py` first (1–2 min).

**About the download.** The URL is CUB's CaltechDATA record (1.2 GB). The md5 in
`cub_meta.CUB_MD5` is the one CaltechDATA publishes on the record page, checked
on 23 Sept 2026. A mismatch can only mean a corrupt download, so the archive is
renamed to `.corrupt` and the run stops; rerunning fetches it again. Beyond that
there is a structural gate: all 11,788 Waterbirds images must be found in
`images.txt`, every (image, attribute) and (image, part) pair must appear exactly
once, and every image must have a box. If the download itself fails,
`cub_meta.py` prints the manual `curl` + `tar` command.

## 5. How to read the results, in this order

1. `results/v6_cub_meta.md`: the verdict must be OK.
2. `results/v6_confirm_check.md`: must say **AGREEMENT OK**. Otherwise the screen
   does not measure what `group_margins.py` measures, and nothing else can be read.
3. `results/v6_sv_screen.md`: read `|S|` and the floor at the top, then the
   Answer, then the depletion table. For any hit, also check `AUC place` and
   `AUC old g`. A hit that is really the old background group in disguise is not
   independent of sessions 1–4.
4. `results/v6_replication.md`: only meaningful if there is a hit.

## 6. The outcomes

1. **A hit that replicates on test.** This is a real image property whose group
   sits entirely off the margin, and it is the first candidate for the alpha > 1
   branch on real data. The next step is `long_horizon.py` on that bundle, to
   test the alignment identity's prediction for beta. That needs the val/test
   bundles regrouped the same way; the candidate is defined for all splits, so
   this needs a regrouping call, not new theory. Two things remain open even
   then: Def 3.3 needs `A ≠ B` across the groups, which the screen does not
   check, and the alignment identity is the load-bearing assumption
   (`SESSION4.md` §3).
2. **A hit that does not replicate.** A chance false positive. Holm allows one
   5% of the time.
3. **No hit.** Nothing on the page is a failed experiment: it is the finite-sample
   face of the insight already recorded (`SESSION5.md` §9). With |S| = 421, a
   natural image property essentially never avoids the margin set. Aser's rule
   stands that no negative result goes into the ICLR paper, so this outcome
   supports the theory remark and is not reported as an experiment.

## 7. Known limitations

- **The family is finite.** "No hit" means no hit among ~650 pre-registered
  properties, not "no property exists". It cannot be otherwise without looking at
  the margins, which is circular.
- **CUB attributes are crowd-sourced and noisy.** Noise dilutes a real effect. It
  cannot create a false hit, because the test's null is exact.
- **The graded depletion test has little power for small groups.** In the mini
  check, a planted margin-free group of 90 points gets depletion p = 0.02 but
  p (min) = 1e-45. The hit rule uses p (min), which uses the whole margin
  distribution, not just the count.
- **`.gitignore` bug, FIXED 23 Sept 2026 (Aser's go-ahead).** The line
  `!results/*.csv          # a small summary csv ...` carried a trailing comment.
  Git does not strip that, so the negation matched nothing and
  `results/v5_bird_fraction.csv` never came back on a pull. The comment is now on
  its own line and `git check-ignore` confirms `results/*.csv` is tracked. The
  next push from the box therefore also adds `v5_bird_fraction.csv` (~1.4 MB).

## 8. Run log

**Run 1, 24 Sept 2026 (`results/logs/session6_20260924_031959/`): aborted at
`20_cub_meta`, nothing screened.** Self-checks all passed. Two problems:

1. **The download got HTTP 403 through Python's urllib**, while `curl -L` on the
   same box and the same URL worked. The visible difference is the client (urllib
   sends `User-Agent: Python-urllib/3.x`); the server's exact rule was not
   investigated. Fix: `cub_meta.download()` now uses curl when it is installed and
   falls back to urllib with a curl-like User-Agent. Self-tested on both paths.
   Aser had already fetched and extracted the archive by hand into `data/` with
   the command `cub_meta.py` printed. That layout is found as it is (self-tested
   with the exact `tar` member list), so the rerun does not download at all. When
   the archive sits next to an existing extraction, its md5 is now checked and
   reported in `v6_cub_meta.md`. It cannot block the run, because the extracted
   files are what gets used and the structural checks gate them.
2. **The automatic push failed** ("could not read Username"): a background job
   cannot prompt for a GitHub token. Aser's decision: pushing is removed from the
   runner entirely, and he commits and pushes `results/` by hand (commands in §4).
