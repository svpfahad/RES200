# Submission Info Needed — fill this in, then tell me "done"

**Paper:** *Knowing When to Trust a pKa Model: Conformal Prediction Intervals and an Applicability Domain for XGBoost Descriptor Models*
**File that gets submitted:** `manuscript/submission/JURI_pKa_Uncertainty_AD_Atwi.docx`
**Built by:** `scripts/build_juri_paper.py` (re-run after edits)
**Status:** Fully drafted and formatted to JURI house style. NOT yet submission-ready — the items below are what's missing.

> How to use this file: replace every `FILL: ...` line with your answer, and tick the `[ ]` boxes.
> When you're done, tell me and I'll edit the build script + `CITATION.cff`, regenerate the `.docx`,
> re-verify every number, and produce the final file (plus a PDF and cover letter if you want them).

---

## A. REQUIRED — true blockers (paper cannot go out without these)

### A1. GitHub repository URL  ⛔ the one hard blocker
The paper says the code is "available at **[GitHub URL — FILL BEFORE SUBMISSION]**" and `CITATION.cff` has the same placeholder. A reviewer will try to open this link, so it must be a **public** URL that contains the runnable `sota_pka/uq/` package.

Pick one:
- [ ] **Use the existing repo** `https://github.com/svpfahad/RES200`
      — ⚠️ this also exposes your unpublished QLattice paper, the CO₂ patent package, draft manuscripts, and editor correspondence. Not recommended unless you clean it first.
- [yes] **Create a new, dedicated public repo** with only the `sota_pka/uq/` package + data pointers (recommended — clean, scoped to this paper). I can prep the file list for it.
- [ ] Other:

  > FILL (final public URL to print in the paper): `https://github.com/____________________`

- [ ] I confirm this repo is **PUBLIC** (or will be public before I submit).

### A2. Submission format / portal
What does the JURI portal want uploaded? (Your first paper went through `journalssystem.com/juri`.)
- [ ] `.docx` only
- [ ] Compiled **PDF** as well — I'll generate one from the final `.docx`
- [ ] Both

  > FILL (anything specific the portal asks for, e.g. separate title page, line numbers, anonymized version):

---

## B. STRONGLY RECOMMENDED — finish the self-citation

### B1. Your first (published) JURI paper — full citation
Reference (1) currently reads only: *"Atwi, F. A. & Al-Khater, M. XGBoost-based prediction of pKa values from molecular descriptors. J. Undergrad. Res. Int. (2026)."* — no volume/issue/pages/DOI. Add whatever is now available:

  > FILL Volume:
  > FILL Issue:
  > FILL Pages / article no.:
  > FILL Year:
  > FILL DOI or article URL:

- [ ] Not published yet → leave as "(2026), in press" for now (acceptable, not a desk-reject risk)

---

## C. CONFIRM — decisions baked into the current draft (verify, or tell me to change)

### C1. Authorship  ⚠️ please read
This paper is currently **single-author: Fahad Ali Atwi**, with Dr. Mohammed Al-Khater credited only in the Acknowledgements as mentor.
Your **first, published** paper (JURI-00032-2025) listed **both** of you, with **Al-Khater as the corresponding author**, and this new paper builds directly on that "foundational pKa modeling work." Confirm the intent:
- [ ] Correct — keep it single-author (Atwi), Al-Khater acknowledged only
- [ ] Change — add Dr. Al-Khater as co-author

  > FILL (if co-author: his affiliation, ORCID, email, and who is corresponding):

### C2. Author details (currently in the paper)
- Name: **Fahad Ali Atwi** — [ ] correct
- Affiliation: **Department of Chemical Engineering, KFUPM, Dhahran 31261, Saudi Arabia** — [ ] correct
- ORCID: **0009-0003-7015-9092** — [ ] correct
- Corresponding email: **s202283300@kfupm.edu.sa** — [ ] correct  (or > FILL preferred email: )

### C3. Target venue
- [ ] **JURI** (Journal of Undergraduate Research International) — same as draft
- [ ] Other:

  > FILL (if changing venue, name it — formatting/refs may need rework):

### C4. Title  (confirm or rewrite)
Current: *Knowing When to Trust a pKa Model: Conformal Prediction Intervals and an Applicability Domain for XGBoost Descriptor Models*
- [ ] Keep    > FILL (new title if changing):

### C5. Keywords  (confirm or edit; JURI allows 4–8, currently 6)
Current: *pKa prediction; uncertainty quantification; conformal prediction; applicability domain; XGBoost; QSPR*
- [ ] Keep    > FILL (replacements if any):

---

## D. OPTIONAL — improves the submission, not strictly required

- [ ] **Cover letter** — your first submission included one (`KFUPM_309_covering_letter.doc`). I can draft a fresh one for this paper. → > FILL: yes / no
- [ ] **Suggested / opposed reviewers** — some portals ask. → > FILL names+emails if you have them:
- [ ] **Acknowledged funding/grant number** — currently states "no specific grant." → > FILL if that changed:
- [ ] **Mentor sign-off** — do you want Dr. Al-Khater to review before you submit (as with paper 1)? → > FILL: yes / no

---

## E. What I'll do once you return this (no input needed from you here)

1. Set `REPO_URL` in `scripts/build_juri_paper.py` and `repository-code:` in `CITATION.cff` to your A1 URL.
2. Update reference (1) and `CITATION.cff` `preferred-citation` with the B1 details.
3. Apply any C-section changes (authorship, title, keywords, venue).
4. Re-run `.venv_mac/bin/python scripts/build_juri_paper.py` → regenerates `JURI_pKa_Uncertainty_AD_Atwi.docx`.
5. **Re-verify every number** in Tables 1–5 against `sota_pka/paper_assets/uq/RESULTS_uq.md` (independent recompute, per your "show verification" rule).
6. Produce the final submission file (+ PDF and cover letter if you ticked them).
7. Give you a one-line "ready to upload" confirmation with the exact file path.

---

### Quick reference — the only edits that touch the manuscript text
| Item | Where it lives now | Needs |
|---|---|---|
| GitHub URL | `build_juri_paper.py` line ~30 `REPO_URL`; `CITATION.cff` `repository-code` | **A1** |
| Prior-paper DOI | `build_juri_paper.py` `REFS["atwi"]`; `CITATION.cff` `preferred-citation` | **B1** |
| Authorship | `build_juri_paper.py` author block + Acknowledgements | **C1** |
