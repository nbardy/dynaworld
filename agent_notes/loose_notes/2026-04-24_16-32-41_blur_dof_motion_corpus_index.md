# Blur / DoF / Motion Corpus Index Session

Date: 2026-04-24

Trigger:

User asked to keep going: find more papers, download PDFs, extract text, and
make a paper index for blur / depth-of-field / motion blur in NeRF and 3DGS.

Actions:

- Built local corpus folder:
  `research_notes/blur_dof_motion_papers/`.
- Downloaded 55 PDFs into `research_notes/blur_dof_motion_papers/pdfs/`.
- Extracted 55 text files with `pdftotext -layout` into
  `research_notes/blur_dof_motion_papers/text/`.
- Generated `research_notes/blur_dof_motion_papers/metadata.json`.
- Generated `research_notes/blur_dof_motion_papers/paper_index.md` with
  clusters, priorities, source links, and local PDF/text links.
- Added `research_notes/blur_dof_motion_papers/extraction_notes.md` with
  formula anchors for defocus, camera motion blur, dynamic object blur, and
  event/spike/IMU assistance.
- Updated routing docs:
  `research_notes/README.md`,
  `research_notes/potential_directions_index.md`, and
  `research_notes/blur_dof_motion_paper_review.md`.

Operational notes:

- The first downloader stalled because `curl` progress output filled a captured
  stderr pipe. It was killed, preserving the partial downloads, then rerun with
  silent curl output and existing-file skipping.
- Final corpus size was about 793 MB.
- Existing unrelated dirty worktree files were left untouched.

Current next step:

Do not add more bibliography first. Extract formula/loss/dataset tables from
the A-priority papers and use them to design a synthetic exact-sampling renderer
test before implementing production blur support.
