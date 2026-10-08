# hGRU Gabor initialisation

The hGRU baseline uses the Gabor initialisation file released with the reference
implementation by Linsley et al. (NeurIPS 2018). The third-party binary is not
redistributed in this package.

Obtain `weights/gabors_for_contours_7.npy` from the following pinned upstream
revision:

<https://github.com/serre-lab/hgru_share/blob/4ac92bd12b3c91092415ee78530f1e4a81c2f2c0/weights/gabors_for_contours_7.npy>

Place the file at:

```text
models/assets/gabors_for_contours_7.npy
```

The required SHA-256 digest is:

```text
4f0482e5c032d0c52fea89ca24cbe5d059bddf94ca7de9ad2dd826a968ecd0f4
```

The model verifies this digest before loading the pickled NumPy dictionary and
rejects missing or modified files. No automatic download or fallback
initialisation is performed.
