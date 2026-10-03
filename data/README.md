# FlyGPT data directory

FlyGPT v0.8.0 no longer needs local FlyWire v783 Parquet parts for connectome
queries. Janelia MaleCNS v1.0 is queried through neuPrint using
`NEUPRINT_TOKEN`.

This directory is still used for local runtime state such as:

- `flygpt_auth.sqlite3`
- `flygpt_memory.sqlite3`

Those databases remain ignored by Git.
