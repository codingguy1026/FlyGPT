# FlyWire data directory

Place the split v783 Parquet files in `data/flywire_parts/`.

The expected default filenames are:

```text
proofread_connections_783_part_01.parquet
...
proofread_connections_783_part_09.parquet
```

The Parquet files are not committed to Git because they are large. The loader
also accepts a single Parquet file or a glob path.
