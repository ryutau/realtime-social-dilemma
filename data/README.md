# Data

The files in `source/` contain participant data before completion exclusions.
Analysis scripts apply these exclusions when loading data through
`src.common.load_pd`. Column definitions are in the [data dictionary](dictionary.md).

## Analysis samples

The main experiment has 561 eligible participants and 280 complete pairs.
Participants who did not complete the post-game tasks are excluded. The
completed partner of one excluded SM participant remains in individual-level
analyses; pair-level analyses use only complete original pairs.

The pilot has 152 eligible participants and 76 pairs. Participants who did not
complete the post-game tasks and their partners are excluded.

Among eligible participants, Main risk and ambiguity measures are missing for
91 SM participants because of a technical issue. Two Pilot SM participants have
missing age. Regressions
use complete cases; these missing measures do not exclude participants from
PD analyses.

The inputs include derived SVO and risk/ambiguity measures. Their calculation
from raw task responses is outside this repository.
