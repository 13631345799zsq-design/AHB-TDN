# Data preprocessing

These 13 scripts are visible copies of the processing programs distributed with the experimental database. Each dataset directory retains its original `0_progress_program/` structure. Input and output paths point to `Database/` beside the numbered repository directories, independent of the current working directory.

The same scripts remain available in the release database for standalone use. Use one location consistently; both versions operate on the same database files.

## Processed data

The 200 distributed `.pkl` samples can be loaded directly from `Database/0_Dataset/`; preprocessing is not required before training on these samples.

## Optional regeneration from raw records

Run the scripts for the desired dataset in this order:

1. `1_Resampling.py`: align paired heater and temperature CSV records and write `dataset_aligned.csv` in each selected `Command_*` directory.
2. `3_downsampling.py`: write aligned records at the configured larger time intervals, including `dataset_aligned_0.5s.csv`.
3. `2_point_data_complex.py`: construct normalized modeling samples from the aligned 0.5-second records and write `.pkl` files into `Database/0_Dataset/`.

For example, from this directory:

```bash
python "1_Multi_condition_sequence_dataset_[prototype_1]/0_progress_program/1_Resampling.py"
python "1_Multi_condition_sequence_dataset_[prototype_1]/0_progress_program/3_downsampling.py"
python "1_Multi_condition_sequence_dataset_[prototype_1]/0_progress_program/2_point_data_complex.py"
```

Before regeneration, review `AT_FOLDERS` in the alignment/downsampling scripts and `BASE_PATH`, `SAVE_PATH`, `SOURCE_CSV`, and `SAMPLE_INTERVAL` in the sample-construction script. The original experiment selections are retained. For prototype 1 multi-condition data, the sample-construction default is `AT_45` and `Data_AT45_D1_05`; change both paths together to process another ambient-temperature condition. Only process folders present in the downloaded archive.

These programs write generated files into the extracted database. Sample construction can overwrite existing `.pkl` files in its selected output directory; use a separate output directory when comparing regenerated samples with the distributed data.

`4_Repeatability test.py`, supplied for prototype 1 multi-condition data, analyzes aligned trajectories across ambient-temperature conditions and additionally requires scikit-learn.
