# Matched-budget stage-two replay

Only stage two is solved again. Saved projected endpoints and available original stage-one routes are reused.
Every stage-two budget matches that case's stage-one per-call path/trial cap.
Stitched total = previous method total minus previous stage-two solve time plus newly measured stage-two solve time.
It is a timing estimate combining saved and replayed components, not a new end-to-end measurement.

| Method | Budget in both stages | Final cost | New stage two (s) | Stitched total (s) | Retained stage-one route | Status |
| --- | --- | ---: | ---: | ---: | --- | --- |
| 3R independent | 50/75 | 0.3818392430 | 4.254 | 77.401 | True | success |
| 3R independent | 100/150 | 0.3817177694 | 2.911 | 85.301 | True | success |
| 3R independent | 200/300 | 0.3817177694 | 3.410 | 103.543 | True | success |
| 3R independent | 300/450 | 0.3817177694 | 3.426 | 120.665 | True | success |
| 3R independent | 400/600 | 0.3817177679 | 3.468 | 139.512 | True | success |
| 4R independent | 50/75 | 0.1306891535 | 6.220 | 100.749 | True | success |

Source JSON hashes, graph identity checks and all replay validations are saved per case.
No stage-one GCS, sampled-goal or nonlinear-goal experiment ran in this replay.
Any unavailable original stage-one route is explicitly recorded; a previous stage-two route is never used as a substitute.
The separate unfinished 4R first-stage sweep remains paused.
