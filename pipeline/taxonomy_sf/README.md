# SkillsFuture taxonomy as the front end and back end key it

Mapping used by the product (GreyGigz term to SkillsFuture term):

| GreyGigz / front end | SkillsFuture | File | Rows |
|---|---|---|---|
| Category | Sector | `sector.csv` | 39 |
| Specialisation | Track | `track.csv` (`sector_id`) | 247 |
| Skills | TSC (technical skills and competencies) | `tsc.csv` | 2,088 |
| (job role) | Job role | `job_role.csv` (`track_id`) | 2,001 |
| (role to skill link) | | `job_role_tsc.csv` | 43,958 |

A `search_tags` entry is `"<track> (<sector>)"`. A track name alone is ambiguous (9 names recur across sectors); the pair is unique.

`tsc.zip` is the archive as received. It also holds two files nothing reads yet: `job_role_profile.csv` (role descriptions and
critical work functions) and `tcs_descriptions.csv` (TSC descriptions by sector and category).

## These IDs are not the IDs in `../taxonomy_greygigz/`

The content is identical (same 247 track and sector pairs, same 2,088 TSC titles, same 2,001 role-to-TSC sets), the integer IDs are
not: no track ID agrees and one TSC ID agrees. The predicted-tag files (`../data_sat/tags_*.json`) use the old IDs. Translate
through `taxonomy_sf.bridge_to_greygigz()` (by TSC title and by track and sector name); never compare IDs across the two.
