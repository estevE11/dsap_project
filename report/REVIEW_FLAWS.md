# Paper Review - Flaws to Fix Before Delivery

## ✅ Resolved Issues

### 1. Wiener Filter Description
**Status:** CORRECT - The Toeplitz/STFT description is accurate. Implementation was done in MATLAB by a colleague (not in this Python repo).

### 2. RLS References in Abstract/Contributions
**Status:** FIXED - Removed RLS references. Only 2 algorithms are compared: Multichannel Wiener Filter and MLE (Das et al.).

### 3. Missing Line Break
**Status:** FIXED - Added `\\` before Limitations paragraph.

### 4. Room Dimensions
**Status:** FIXED - Clarified that original paper only specifies 2D (5×5 m), we added 3 m height as a realistic assumption.

### 5. Simulation Code Bug
**Status:** IGNORED - Not relevant for the paper.

### 7. Terminology: Wiener Filter Naming
**Status:** FIXED - Now uses "Multichannel Wiener Filter (MWF)" consistently throughout.

### 8. Dataset Naming
**Status:** FIXED - Clarified as "EnsembleSet dataset (BBC Symphony Orchestra 'Misero Pargoletto' recording)".

### 9. RLS in Conclusion
**Status:** FIXED - Changed "block-wise RLS" to "adaptive online updates".

### 10. Code Names in Text
**Status:** FIXED - Removed `stft_v1`, `mle_lambda0_paper`, `mcwf_v1` from paper text.

### 11. "SIR is infinite" Claim
**Status:** FIXED - Changed to "numerically unbounded" with explanation about perfect inversion leaving no residual interference.

### 12. Capitalization
**Status:** FIXED - Using "MWF" consistently avoids the "blind/Blind" issue.

### 13. Diagonal Constraint Reference
**Status:** FIXED - Added explanation: "the diagonal unity constraint ($H_{nn}=1$)".

---

## 🟡 Pending Issues

### 6. Table I Metrics Inconsistency
**Location:** Lines 103-117

**Problem:** Compares different metrics:
- Our Implementation: OPS and **SDR**
- Reference (Das et al.): OPS and **APS**

**Status:** DEFERRED - Will be corrected later.

---

## Checklist

- [x] ~~Fix Wiener filter description~~ (Was correct - MATLAB implementation)
- [x] Remove RLS references from abstract/contributions (Fixed)
- [ ] Fix Table I metrics consistency (Deferred)
- [x] Add `\\` before Limitations paragraph (Fixed)
- [x] Clarify room dimensions (Fixed)
- [x] ~~Fix `max_order` bug~~ (Ignored)
- [x] Standardize Wiener filter terminology → "MWF" (Fixed)
- [x] Clarify dataset naming (Fixed)
- [x] Remove RLS mention from Conclusion (Fixed)
- [x] Remove code names from text (Fixed)
- [x] Clarify "SIR is infinite" claim (Fixed)
- [x] Fix capitalization of "blind" → using "MWF" (Fixed)
- [x] Clarify "diagonal constraint" reference (Fixed)
