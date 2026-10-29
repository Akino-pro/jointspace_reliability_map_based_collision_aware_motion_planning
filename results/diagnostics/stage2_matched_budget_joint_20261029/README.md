# Matched-budget stage-two replay

Only stage two is solved again. Saved projected endpoints and available original stage-one routes are reused.
Every stage-two budget matches that case's stage-one per-call path/trial cap.
Stitched total = previous method total minus previous stage-two solve time plus newly measured stage-two solve time.
It is a timing estimate combining saved and replayed components, not a new end-to-end measurement.

| Method | Budget in both stages | Final cost | New stage two (s) | Stitched total (s) | Retained stage-one route | Status |
| --- | --- | ---: | ---: | ---: | --- | --- |
| 3R joint | 50/75 | 0.3861941211 | 9.555 | 17.124 | False | success |
| 3R joint | 100/150 | 0.3850137378 | 8.706 | 15.714 | False | success |
| 3R joint | 200/300 | 0.3850137377 | 8.897 | 16.788 | False | success |
| 3R joint | 300/450 | 0.3850137377 | 9.201 | 18.078 | False | success |
| 3R joint | 400/600 | 0.3850137377 | 9.595 | 19.448 | False | success |

Source JSON hashes, graph identity checks and all replay validations are saved per case.
No stage-one GCS, sampled-goal or nonlinear-goal experiment ran in this replay.
Any unavailable original stage-one route is explicitly recorded; a previous stage-two route is never used as a substitute.
The separate unfinished 4R first-stage sweep remains paused.
