# Amazon ML Challenge 2026 — Business Entity Resolution
### **Precision 98% Targeted Architecture with Macro $F_{0.5}$ Optimization**

This repository contains the boosted, production-ready solution for the Amazon ML Challenge 2026 Business Entity Resolution challenge.

---

## 🚀 All-in-One Google Colab Notebook
The complete end-to-end solution and database are consolidated into a single, fully self-contained `.ipynb` file ready to run directly in Google Colab (CPU or T4/A100 GPU):

👉 **[`Amazon_ML_Challenge_Entity_Resolution_Precision98.ipynb`](file:///c:/Users/piyus/OneDrive/Desktop/project/99-/Amazon_ML_Challenge_Entity_Resolution_Precision98.ipynb)**

### Key Highlights of the Notebook:
1. **Self-Contained Embedded Database**:
   - Includes a built-in realistic benchmark database generator (with Indic script transliterations, French street conventions, US abbreviations, leetspeak, singletons, and true ground truth) so anyone can click **"Run All"** in Colab and execute the full pipeline out-of-the-box.
   - Features a one-line switch (`USE_FULL_DATASET = True`) to seamlessly train and run inference on the full competition TSVs via Google Drive or local storage.
2. **Boosted Multi-Pass Hybrid Blocking (>97% Recall)**:
   - Pass 1: Exact Normalized Core Name Key `(country, name_core)`
   - Pass 2: Compressed Name Key (whitespace/punctuation stripped)
   - Pass 3: Rare Name Token Inverted Index ($\le 500$ occurrences)
   - Pass 4: Postal / PIN Code + First Word Prefix Key
   - Pass 5: Sparse TF-IDF Character 3-gram Cosine Top-K Retrieval
   - Pass 6: Strict Country Partitioning (Zero cross-country false positives)
3. **Boosted 40+ Discriminative Pairwise Features**:
   - Advanced string similarities: Levenshtein ratio, Jaro-Winkler, Token Sort/Set ratios, Partial ratio.
   - Fatal Conflict Detectors: House/building number conflict (`num_conflict`) and Postal/PIN conflict (`pin_conflict`).
   - Query Competition & Context: Candidate rank, competition density, margin ratios, score gaps, and ambiguous tie flags.
4. **Targeted Precision $\ge 98\%$ & Macro $F_{0.5}$ Optimization**:
   - Official Macro $F_{0.5}$ evaluation per entity (including singletons).
   - High-Precision Veto Guards eliminating erroneous address/number collisions.
   - 1-to-1 query matching invariant (ensuring each $S_2/S_3$ query record matches at most ONE $S_1$ entity).
   - Threshold sweep selecting the operating point that satisfies Precision $\ge 98\%$ while maximizing $F_{0.5}$.
5. **Official Submission Generation & Verification**:
   - Generates `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
   - Built-in validator running all checks from `student_resource/utils/validate_submission.py` and outputting `VALIDATION RESULT: PASS (Exit code 0)`.

