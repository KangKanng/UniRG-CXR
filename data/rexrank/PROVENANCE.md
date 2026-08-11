# IU-Xray split provenance

ReXrank states that its IU-Xray evaluation uses the R2Gen split. The manifests
in this directory contain the same 2,069 train, 296 validation and 590 test
studies as R2Gen's `iu_xray/annotation.json`.

They were verified against the user-provided `../iu_xray.zip` on 2026-07-17:

- ZIP SHA-256: `924bd5ef549c642c29a60e18d42c5882b40d263020eae000540dc7ede7c7f6b6`
- study-ID set differences: 0 for every split
- study-ID order differences: 0 for every split
- report-text differences: 0 across all 2,955 studies

The existing JSONL manifests are retained because they are semantically
identical and much smaller than the 1.1 GB image archive.
