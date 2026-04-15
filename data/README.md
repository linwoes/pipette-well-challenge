# Pipette Well Challenge Dataset

This directory contains the pipette well dataset for the challenge.

## Large File Storage

The `pipette_well_dataset.tar.gz` file is tracked using Git LFS (Large File Storage).

### Downloading the Dataset

The dataset is stored in this repository using Git LFS. To retrieve the actual dataset file:

1. **If you have `git-lfs` installed:**
   ```bash
   git lfs pull
   # or
   git lfs pull --include="*.tar.gz"
   ```

2. **If you don't have `git-lfs` installed:**
   
   Use the provided download script:
   ```bash
   python data/download_dataset.py
   ```

### Extracting the Dataset

Once you have the `pipette_well_dataset.tar.gz` file:

```bash
tar -xzf pipette_well_dataset.tar.gz -C data/
```

This will extract the dataset into `data/pipette_well_dataset/`.

### Dataset Information

- **File:** `pipette_well_dataset.tar.gz`
- **Size:** ~166 MB (174,132,820 bytes)
- **SHA256:** `88b8799cee91f72f9eebae4b81625f1ef0aee2ae8680ffdbce743d29f99e01cc`
- **Format:** Compressed tar archive

For more information about the challenge and dataset structure, see the main README.md.
