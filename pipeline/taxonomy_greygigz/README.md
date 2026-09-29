# SkillsFuture taxonomy in the GreyGigz schema

Raw MySQL dumps plus side CSVs, produced by `skillsfuture_to_greygigz.py` (a separate script,
not in this repo) from `jobsandskills-skillsfuture-skills-framework-dataset.xlsx`. Generated
2026-09-29. Treat everything here as read-only input; regenerate rather than hand-edit.

This folder is a **taxonomy**, not a gig/provider corpus, so it is not a `--data-dir` dataset.
Read it through `pipeline/greygigz.py`, which parses the dumps directly (no MySQL needed, and
nothing here is ever loaded into a live database). `python pipeline/greygigz.py` prints a summary
and fails if any count below is off.

| File | Table / content | Rows |
|---|---|---|
| `categories.sql` | `categories` = SkillsFuture **Track** | 247 |
| `specialities.sql` | `specialities` = **Job Role**, with descriptions | 2,001 |
| `tags.sql` | `tags` = TSC/CCS skill titles | 2,088 |
| `speciality_tags.sql` | role-skill bridge (43,958 distinct pairs, no dangling references) | 43,958 |
| `services.sql` | `services`; every category has `service_id` 1 | 4 |
| `speciality_tag_levels.csv` | proficiency per bridge row, one row per bridge pair | 43,958 |
| `sector_by_category.csv` | `category_id,sector,track`. Sector was dropped from the schema on purpose | 247 |
| `truncated_names.csv` | `field,original,stored` for values cut to fit their column | 1 |

## Measured properties (from `greygigz.py`, pinned by `tests/test_greygigz.py`)

- **Tags per role:** min 2, median 20, mean 22.0, max 78. Every tag is used by at least one role.
- **Generic tags exist.** "Stakeholder Management" is on 1,060 of 2,001 roles, "Change Management"
  on 672, "Continuous Improvement Management" on 586.
- **Identical roles.** 619 roles have exactly the same tag set as at least one other role
  (1,606 distinct tag sets for 2,001 roles). Score role retrieval at tag-set-equivalence level
  (`Taxonomy.equivalence_classes`), not by exact role ID.
- **Track names repeat.** 234 distinct names over 247 category IDs: 9 names are shared by 22 IDs
  (13 surplus rows), each in a different sector, e.g. "General Management" on five IDs. Key
  tracks by `category_id`, never by name.
- **One truncated track name:** category 80 (sector "Energy and Power"). The stored name is 234
  characters, the original 366; `Taxonomy.track_display_name` restores it. The name is an
  11-item " / "-joined list rather than a single track title.
- **Levels.** The Basic/Intermediate/Advanced to 2/4/6 mapping applies to the 346 CCS rows
  (`tsc_ccs_type=ccs`). The other 43,612 rows are TSC and keep native levels 1-6 (1: 1,300,
  2: 4,971, 3: 11,264, 4: 13,588, 5: 9,939, 6: 2,550), so levels are not on one scale.
  `raw_levels` can hold several values (574 rows); `level` is their maximum.
- **Line endings.** The files are stored with LF in git but check out as CRLF on Windows
  (`core.autocrlf=true`). The loader is newline-agnostic; do not rely on byte sizes or raw hashes.
