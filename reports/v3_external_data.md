# v3 Phase C — external reference data (research only)

Date checked: 2026-10-27. No downloads, registrations or emails were made. "Not confirmed" means
the search or fetch did not show it; it does not mean it is false. FFR values from TwinLoop remain
unvalidated research output.

**Why this report exists:** Gate G1 failed (reports/v3_A3_definition.md). Under every smoothing
definition σ_s ∈ {0…2} mm, at most 4.8 % of moderate+ GT lesions survive a σ 0.5 mm blur of the GT
mask, so the ImageCAS GT-mask stenosis "ruler" cannot score a segmentation fix. An independent
reference (QCA, invasive FFR or expert grades) is needed.

| Candidate | Reference it has | Cases | Licence / access | Availability checked | URL |
|---|---|---|---|---|---|
| Rotterdam Coronary Artery Stenosis Detection and Quantification framework (Kirişli et al., Med Image Anal 2013) | Reference standard for stenosis detection and quantification (QCA-based per the framework description; exact per-lesion grades not confirmed here) | 48 CTA (18 training, 30 test, per a later study) | Not confirmed. Paper says data are made available via the project site and it is open to submissions | UNCERTAIN. Project site `coronary.bigr.nl/stenoses` could not be fetched: TLS certificate is for `*.gitlab.io`, not `coronary.bigr.nl` (2026-10-27). Not confirmed whether the data are still downloadable | https://repub.eur.nl/pub/41673 ; https://busqueda.bvsalud.org/portal/resource/es/mdl-23837963 |
| ICADD (Bohra et al., medRxiv 2023, JNCASR, South India) | CCTA + invasive angiography + invasive FFR (+ echo, haematology) | 70 patients (70 CCTA, 70 ICA with FFR) | Not confirmed: the Data Availability section could not be read (full-text PDF returned HTTP 403) | Not confirmed. A JNCASR repository record exists | https://www.medrxiv.org/content/10.1101/2023.05.13.23289934 ; https://libjncir.jncasr.ac.in/xmlui/handle/123456789/3132 |
| CT-FFR computational benchmark (four methods vs invasive FFR) | Invasive FFR; clinical data said to be freely downloadable on figshare | 29 subjects (CCTA before invasive angiography) | Said to be open on figshare; DOI not retrieved (page returned HTTP 403) | Not confirmed; find the DOI from the paper | https://www.researchgate.net/publication/334527329_Non-invasive_coronary_CT_angiography-derived_fractional_flow_reserve_A_benchmark_study_comparing_the_diagnostic_performance_of_four_different_computational_methodologies |
| ASOCA (MICCAI 2020) | Expert coronary segmentation masks, centrelines, calcium scores. No per-lesion stenosis labels found | 40 train (20 healthy, 20 CAD) + 20 test | UK Data Service (terms not confirmed) | Listed as available on the challenge site | https://asoca.grand-challenge.org/ ; https://link.springer.com/article/10.1038/s41597-023-02016-2 |
| ImageCAS-X (2026 preprint) | Segmentation + centreline benchmark; functional/QCA labels not mentioned in results | not checked | not checked | Not confirmed | https://arxiv.org/pdf/2608.30404 |
| MultiD4CAD | CAD diagnosis, fat segmentation; FFR not confirmed | 118 CCTA | Data use agreement | Zenodo record exists | https://zenodo.org/records/15148653 |
| CADICA | Invasive angiography images with lesion classes (not CT) | n/a for CCTA | public | Not CCTA, so no use for this pipeline | https://itis.uma.es/en/cadica-an-advancement-in-coronary-artery-disease-diagnosis-with-over-6-000-accessible-images/ |

Catalog check: a curated list of open cardiovascular datasets lists no CCTA set with FFR, QCA or
stenosis grades (ImageCAS, ImageALCAPA, 3D CAS Images, MultiD4CAD; segmentation or CAD diagnosis
tasks only): https://github.com/gfahrni/open-cardiovasc-imaging-datasets . A scoping review of 38 open
cardiovascular imaging datasets exists and was not read:
https://www.sciencedirect.com/science/article/pii/S3050577126000393 .

## Assessment

- No public CCTA set with a stenosis reference at scale was found. The three candidates with an
  independent reference (Rotterdam: QCA-style reference; ICADD: invasive FFR; the CT-FFR benchmark:
  invasive FFR) are small (29 to 70 cases), so they could support a feasibility check, not a
  statistically strong test.
- Next steps for the owner (nothing was done): (1) read ICADD's Data Availability section and
  contact the authors if access is on request; (2) find the figshare DOI of the CT-FFR benchmark;
  (3) contact Erasmus MC about the Rotterdam framework data, since its website did not load.
- Not searched: commercial or consortium cohorts (e.g. trial data), which would need agreements.
