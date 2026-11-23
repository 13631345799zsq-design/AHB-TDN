# AHB-TDN: Supplementary Code and Experimental Data

This repository provides the firmware, operating-command collection, data-acquisition software, thermal-modeling programs, and terminal PI/MPC controllers accompanying the AHB-TDN research on transient air cooling and dynamic thermal management of multi-heat-source electronic devices.

The experimental database is distributed separately as **`Database.rar` in [GitHub Releases](https://github.com/13631345799zsq-design/AHB-TDN/releases)**. The corrected code layout documented below is available on the `main` branch. See Section 8 for the distinction between the current code and the original `v1.0.0` source archives.

See [Supplementary Materials.pdf](Supplementary%20Materials.pdf) for the supplementary descriptions, figures, tables, controller-tuning results, and model architectures.

## 1. Repository and release layout

The Code section contains the five numbered directories and the supporting documents below. After downloading and extracting the database, the local directory layout is:

```text
AHB-TDN/
|-- 01_STM32_Firmware/          # STM32 firmware and Keil project
|-- 02_Command/                 # Command generator and 120 operating-command CSVs
|-- 03_Date_Software/           # Desktop data-acquisition and PI/MPC software
|-- 04_Modeling/                # Model definitions, training, evaluation, and analysis
|   |-- 1_AHB_TDN_Hidden_State_Sensitivity/
|   |-- 2_LPTN_Heatsink_Point_Sensitivity/
|   |-- 3_Internal_Ablation/
|   |-- 4_External_Model_Comparison/
|   `-- Data_Preprocessing/     # Data-processing scripts retained in Code
|-- 05_Terminal_Program/        # PI, AHB-TDN-MPC, and LPTN-MPC terminal programs
|-- Database/                  # Download separately from Releases
|   |-- 0_Dataset/             # 200 processed modeling samples
|   |-- 1_Multi_condition_sequence_dataset_[prototype_1]/
|   |-- 1_Multi_condition_sequence_dataset_[prototype_2]/
|   |-- 2_Single_condition_sequence_dataset_[prototype_1]/
|   `-- 2_Single_condition_sequence_dataset_[prototype_2]/
|-- requirements.txt
|-- Supplementary Materials.pdf
`-- README.md
```

The database is supplied as a Release attachment rather than as files in the Code section. After extraction, keep the local `Database/` folder beside the five numbered directories. The name `03_Date_Software` is retained from the original project and refers to data-acquisition software.

## 2. Download and setup

1. Download or clone the current `main` branch for the corrected directory structure shown above. Record the code commit SHA when reporting results. See Section 8 before using the original `v1.0.0` source archives.
2. Open [Releases](https://github.com/13631345799zsq-design/AHB-TDN/releases) and download the `Database.rar` attachment from the experimental-database release. The automatically generated **Source code** archives do not contain the database.
3. Extract `Database.rar` with a RAR-compatible archive utility into the repository root. The archive contains the top-level `Database/` folder; avoid nesting it as `Database/Database/`.
4. Confirm that `Database/0_Dataset/Data_AT40_D1_05/1.pkl` exists.
5. Install the dependencies needed for the selected task.

```bash
python -m pip install -r requirements.txt
```

The requirements file lists the combined dependencies of the supplied programs; it is not a pinned environment lock file. Modeling primarily uses NumPy, PyTorch, and Matplotlib. Command generation and data preprocessing additionally use Pandas and SciPy; repeatability analysis uses scikit-learn. The desktop application additionally requires PyQt5, pyqtgraph, pyserial, and pymodbus. Terminal PI control only requires pyserial and pymodbus; terminal MPC also uses NumPy, SciPy, and PyTorch.

Run each entry point from its own directory unless stated otherwise. Dataset paths in the supplied training, testing, physical-consistency, and preprocessing scripts are resolved from the script location to the extracted database. Paths therefore do not depend on the original author's drive letters or cloud workspace. Hardware addresses, device settings, experiment selections, and checkpoint selections remain configurable in the original programs.

## 3. Experimental database

The database contains **400 raw acquisition CSV files** and **200 processed `.pkl` samples**. Raw records are stored under `Command_*` folders as paired `cmd_result_*_heater.csv` and `cmd_result_*_temp.csv` files. The archive also retains the original preprocessing-script locations.

| Processed dataset in `Database/0_Dataset/` | Sample count | File IDs | Coverage |
| --- | ---: | --- | --- |
| `Data_AT40_D1_05` | 100 | `1.pkl`-`100.pkl` | Multi-condition sequences, prototype 1, 40 deg C ambient |
| `Data_AT35_D1_05` | 20 | `81.pkl`-`100.pkl` | Multi-condition sequences, prototype 1, 35 deg C ambient |
| `Data_AT45_D1_05` | 20 | `81.pkl`-`100.pkl` | Multi-condition sequences, prototype 1, 45 deg C ambient |
| `Data_AT40_D2_05` | 20 | `81.pkl`-`100.pkl` | Multi-condition sequences, prototype 2, 40 deg C ambient |
| `Data_T2S_D1_05` | 20 | `1.pkl`-`20.pkl` | Single-condition sequences, prototype 1 |
| `Data_T2S_D2_05` | 20 | `1.pkl`-`20.pkl` | Single-condition sequences, prototype 2 |

The `_05` suffix denotes the 0.5-second sampling interval used by the supplied processed datasets. The model data loaders use these dictionary fields:

| Field | Meaning |
| --- | --- |
| `time_list` | Time coordinates |
| `flux_list` | Heater inputs |
| `cooling_list` | Fan inputs |
| `envtemp_list` | Ambient temperature |
| `temp_steady_list` | Steady-temperature data |
| `temp_list` | Measured temperature trajectories |

Use the selected loader and preprocessing code as the reference for units, normalization, and tensor dimensions. Prepared `.pkl` files can be used directly; regenerating them from the raw CSV files is optional. The visible copies of the 13 preprocessing scripts and their execution order are documented in [Data_Preprocessing](04_Modeling/Data_Preprocessing/README.md).

## 4. Hardware firmware and operating commands

### STM32 firmware

`01_STM32_Firmware/` contains the STM32F103C8 firmware and Keil project for 12 heater-power channels, two fan outputs, serial communication, and fan-speed feedback.

- `main.c`: principal firmware source.
- `project.uvprojx`: Keil project.
- `RTE/`: device configuration and startup files.
- `Objects/project.hex`: compiled firmware image.

Firmware development requires Keil MDK and the device support declared in the project, including `Keil.STM32F1xx_DFP.2.4.1`. Check pin assignments, serial settings, I2C addresses, and timers against the actual board before use. Compiler intermediate files, listings, and build reports are retained with the Keil project. They are generated build artifacts; the firmware source, project configuration, and compiled HEX image are also included.

### Operating-command collection

`02_Command/` contains `Command_LHS_Generator.py`, **20 steady-command CSVs**, and **100 transient-command CSVs**. Commands specify 12 heater-power channels and two fan-speed channels.

To generate new commands, run from `02_Command/`:

```bash
python Command_LHS_Generator.py
```

Review `GENERATION_MODE`, command count, duration, sampling interval, channel ranges, and segment-duration settings before generation. Use the supplied command files when the exact distributed sequences are required; running the randomized generator creates a new collection and may write into the existing output directories.

## 5. Data-acquisition software

`03_Date_Software/` provides the PyQt5 desktop application for STM32 serial communication, GL860A temperature acquisition over Modbus/TCP, visualization, command execution, and PI/MPC control. The system acquires 12 heat-source temperatures and one ambient-temperature channel.

From `03_Date_Software/`, run:

```bash
python _1_thermal_control_app.py
```

Configure the serial port and data-logger address for the experimental platform. The desktop MPC program includes its matching `models_1/model_epoch_186.pth` checkpoint. `Control_command.csv` provides an example command.

## 6. Thermal modeling, training, and evaluation

`04_Modeling/` organizes the modeling code into four experiment groups and a data-preprocessing directory:

| Directory | Contents |
| --- | --- |
| `1_AHB_TDN_Hidden_State_Sensitivity/` | AHB-TDN hidden-state sensitivity experiments |
| `2_LPTN_Heatsink_Point_Sensitivity/` | LPTN heatsink-point sensitivity experiments |
| `3_Internal_Ablation/` | Internal ablation experiments |
| `4_External_Model_Comparison/` | Comparisons of MLP, GRU, 1D-CNN, NODE, and AHB-TDN |
| `Data_Preprocessing/` | Raw-data alignment, downsampling, sample construction, and repeatability analysis |

Typical model directories contain a model-definition file, `_1_TRAIN_program.py`, and `_2_TEST_program.py`, with additional parameter-count, timing, plotting, or physical-consistency scripts where supplied.

`4_External_Model_Comparison/` contains five model subdirectories: `1_MLP_int/`, `2_GRU_int/`, `3_1D-CNN_int_64/`, `4_NODE/`, and `5_AHB-TDN/`. Each includes model, training, testing, parameter-count, and CPU-timing programs, together with CPU-timing summary and per-run CSV files. The comparison directory also contains `Data_Record_CPU.xlsx` and `Data_Record_MeanAE.xlsx` as summary records.

For example, from `04_Modeling/4_External_Model_Comparison/5_AHB-TDN/`:

```bash
python _1_TRAIN_program.py
python _2_TEST_program.py
```

Training uses `Database/0_Dataset/Data_AT40_D1_05` by default. Each script retains its original data-volume, fold, epoch, and device settings; inspect these settings before launching a long training run. Training generates model checkpoints and loss/output files in the selected model directory.

Testing requires compatible trained checkpoints. Set the fold and checkpoint selection in the test script to match the available training outputs. The test settings select transient or single-condition sequences (`TF`), prototype (`DF`), ambient temperature (`AT`), and sampling interval (`time_data`). Only the six processed datasets listed above are included.

**No trained `.pth` checkpoints are currently included under `04_Modeling/`.** The three supplied checkpoints belong to the acquisition and terminal-control programs and do not replace the trained parameters required for every modeling experiment. Retrain a selected configuration or supply its matching checkpoint before evaluation.

## 7. Terminal PI and MPC controllers

`05_Terminal_Program/` contains terminal applications for Raspberry Pi or another supported computer connected to the experimental platform.

| Directory | Controller | Matching checkpoint |
| --- | --- | --- |
| `1_PI_Control_terminal/` | PI fan control | Not required |
| `2_MPC_Control_terminal/` | AHB-TDN-based MPC with DBC correction | `models_1/model_epoch_218.pth` |
| `3_LPTN_MPC_Control_terminal/` | LPTN-based MPC with DBC correction | `models_1/model_epoch_289.pth` |

From the desired controller directory:

```bash
python3 main.py --help
python3 main.py --port /dev/ttyUSB0 --ip 192.168.0.2 Control_command_R.csv
```

The serial port and IP address are examples. Check the actual connection settings and command file before execution. Each directory also includes `Control_command_S.csv` and a compiled `project.hex` firmware image. Hardware operation requires the STM32 interface and the temperature data logger.

## 8. Reproducibility and versioning

The database includes the distributed raw acquisition records and six processed datasets; the repository includes the command collection, preprocessing and modeling code, firmware, and runtime controllers.

**Code-version note:** the original `v1.0.0` tag was created before the `04_Modeling/` directory hierarchy was corrected. Its automatically generated **Source code** archives retain the earlier nested layout. Use the current `main` branch with the `Database.rar` attachment for the corrected layout described in this README. Editing `main` or this README does not change the existing tag.

Record both the code commit SHA and the database release tag used for an experiment. For a future fixed code release, check that its tagged source archive follows the directory structure documented here.

The supplied materials provide inputs and procedures for rerunning the covered experiments. Exact reproduction of every manuscript result additionally depends on the experiment-specific settings, trained checkpoints, software environment, and hardware conditions. A complete archive of online closed-loop control recordings is not included in this package.

If `SHA256SUMS.txt` is provided with the release, use its checksum entry for `Database.rar` to verify the download.

## 9. Citation

If you use these materials, cite the accompanying AHB-TDN manuscript and identify the code/data release version. Publication details and a formal citation can be added when available. Preserve applicable third-party notices in firmware and dependency files.
