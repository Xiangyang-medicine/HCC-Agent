# Table 4 title

**Table 4. Formal technical benchmark of the verifier-guided closed-loop Agent system**

# Notes

All quantitative results derive from the complete frozen 4,860-record formal output. The clean primary comparison included 100 cases and three repeated runs per case for both B2 and B4. Both systems used the same language-model backend, frozen prognostic-model tools, assigned evidence passages, and formal cases. B4 additionally used role-specialised planning and synthesis, a deterministic internal verifier, conditional replanning, tool retry, one synthesis revision, and persistent structured state.

The prespecified primary endpoint was the frozen independently implemented composite pass. Confidence intervals were obtained by patient-clustered bootstrap with 2,000 resamples. The B4-versus-B2 p value used a two-sided paired sign-permutation test with 100,000 draws. Ablation p values used the same test with Holm correction across four comparisons.

Exact extractive support requires a generated claim to be an exact sentence from an assigned cited passage. Assigned-passage citation-ID validity requires each citation identifier to belong to the assigned passage set. These automated metrics do not constitute expert assessment of biomedical factuality or semantic retrieval quality. Exact three-run agreement is the proportion of cases with the same binary composite outcome in all three runs.

Planning errors refer to action-specification noncompliance, not disagreement with survival labels. The post-hoc strict report-contract audit changed 0 of 600 clean B2/B4 outcomes and did not replace or modify the prespecified endpoint. The internal verifier is available only in B4 and is not encoded as a zero-valued metric for B2.

Fault-handling differences are B4 minus B2. Each fault used 30 cases and three repeats per system. For the unsupported-request fault, B4 detected the prohibited request and safely exited, but the frozen scoring contract required task completion and therefore classified that terminal outcome as unsuccessful; this mismatch is reported without retrospective correction.

This is a technical Agent benchmark. It does not assess clinical utility, diagnostic accuracy, treatment benefit, deployment readiness, or patient outcomes.
