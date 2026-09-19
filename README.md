[![DOI](https://zenodo.org/badge/1377280318.svg)](https://doi.org/10.5281/zenodo.22848939)

# AssemblyHAR Code

This repository contains the source code used to generate and process the dataset described in the paper:

> **AssemblyHAR: A smartwatch accelerometer dataset for activity recognition and event-log generation in manual assembly**

The code supports the preprocessing of smartwatch accelerometer recordings, the generation of windowed datasets for machine-learning experiments, and the creation of session-level event logs for process-mining analyses.

## Abstract

Data-driven manufacturing increasingly relies on structured process data for process monitoring, analysis, and improvement. However, manual assembly activities are rarely captured as structured event data. This presents a particular challenge in labor-intensive manufacturing systems, where human operators remain central to production activities. Human Activity Recognition (HAR) using wearable sensors provides a means of capturing these activities directly from sensor data. Accelerometers are particularly suitable for this purpose because they capture worker movements while preserving worker privacy and do not rely on environmental conditions such as visibility or illumination. Few existing datasets combine wearable-sensor-based HAR with structured, repeated assembly processes and explicit process-instance information. Here, we present a dataset of manual assembly activities recorded using triaxial accelerometers on both wrists. The dataset comprises 26 experimental sessions with 16 participants performing two structured electric motor assembly processes, resulting in more than 10 hours of sensor data for each motor type. It includes raw accelerometer recordings, annotation and connection logs, labelled and preprocessed sensor data, metadata, event logs in XES format, and windowed datasets. The dataset supports research in human activity recognition, process mining, and data-driven manufacturing in labor-intensive manufacturing systems.

## Dataset Overview

The associated dataset was collected in a controlled laboratory study at the SYDSEN Digital Twins Lab of the Institute of Applied Informatics and Formal Description Methods (AIFB), Karlsruhe Institute of Technology (KIT), in December 2025.

It contains:

- 26 experimental sessions from 16 participants
- Two structured electric motor assembly processes
- Triaxial accelerometer recordings from smartwatches worn on both wrists
- Button-based annotation logs and Bluetooth Low Energy (BLE) connection logs
- Labelled and preprocessed sensor recordings
- Validation reports, statistical summaries, and visualizations
- Event logs in the XES format
- Windowed datasets with grouped and ungrouped activity labels

The smartwatch recordings were collected at a nominal sampling frequency of 100 Hz, with an effective sampling frequency of approximately 80 Hz due to BLE transmission constraints. Video recordings were used exclusively for annotation verification and quality assurance and are not included in the dataset because they cannot be fully anonymized.

## Repository Structure

```text
AssemblyHAR/
├── event_log_generator.py
├── requirements.txt
├── dataset_generator/
│   ├── __init__.py
│   ├── run_dataset_generator.py
│   ├── run_window_overlap_grid.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── activity_manager.py
│   │   ├── dataset_manager.py
│   │   ├── session_processor.py
│   │   ├── signal_preprocessing_config.py
│   │   └── window_generator.py
│   └── helpers/
│       ├── __init__.py
│       ├── constants.py
│       ├── data_labels.py
│       ├── dataset_builder.py
│       ├── dataset_stats.py
│       ├── feature_extraction.py
│       ├── feature_selection.py
│       ├── frequency_estimation.py
│       ├── signal_filtering.py
│       ├── signal_preprocessing_utils.py
│       ├── splits.py
│       ├── wavelet_utils.py
│       ├── window_extraction.py
│       └── window_fusion.py
```

## Main Components

### `dataset_generator/`

The `dataset_generator` package contains the main data-preparation pipeline. It transforms raw session recordings into labelled, preprocessed, and windowed datasets.

Its components are organized as follows:

- **`core/`**: Implements the central processing logic, including session processing, activity management, dataset management, signal-preprocessing configuration, and window generation.
- **`helpers/`**: Provides supporting functionality for signal processing, feature extraction and selection, frequency estimation, activity labeling, dataset construction, statistical analysis, data splitting, and window manipulation.

### `event_log_generator.py`

This script generates one XES event log per experimental session from the activity-segment outputs produced during session processing. It reads the activity-segment CSV files, constructs session-level process logs, and writes a summary JSON file containing aggregated information for the generated event logs.

## Executable Scripts

### `run_dataset_generator.py`

This is the primary entry point for generating the standard windowed datasets.

The script:

- Scans the available session folders under the configured `Data` directory.
- Processes the recorded sensor data and associated annotations.
- Performs the configured preprocessing and window extraction steps.
- Generates raw and feature-based window datasets.
- Creates grouped and ungrouped dataset outputs.
- Writes activity, assembly, and phase summaries.
- Stores processing logs and dataset-related statistics in the output structure under `Processed_Data`.

The input and output locations, as well as processing parameters, should be checked in the script and the corresponding configuration modules before execution.

### `run_window_overlap_grid.py`

This script executes dataset generation for multiple combinations of window sizes and overlap percentages.

The parameter grid is configured through the constants defined at the beginning of the script:

- `WINDOW_RANGE_START_SEC`
- `WINDOW_RANGE_END_SEC`
- `OVERLAP_PERCENTAGES`

The script:

- Iterates over the configured window and overlap combinations.
- Runs the dataset-generation pipeline for each combination.
- Produces the corresponding dataset outputs.
- Reports the success or failure of each parameter combination.

### `event_log_generator.py`

This script is executed separately from the windowed dataset-generation pipeline. It consumes the processed session results and converts activity segmentation information into XES event logs.

The script:

- Searches for processed session folders under `Processed_Data/Processed_Session_Data`.
- Reads the activity-segment CSV file from each session's results directory.
- Generates one XES event log per session.
- Writes event-log files following the session-specific naming convention, such as `session_1_O1_data_event_log.xes`.
- Creates a compact JSON summary for the generated event logs.

## Installation

Install the required Python dependencies using:

```bash
pip install -r requirements.txt
```

The scripts should be executed from the repository's root directory so that the configured relative paths can be resolved correctly.

## Usage Notes

Before running the scripts:

1. Place the required input data in the directory structure expected by the configuration and processing scripts.
2. Check the configured input and output paths.
3. Review the preprocessing and window-generation parameters.
4. Confirm that the required dependencies have been installed.
5. Run the relevant entry point from the repository root.

The exact command-line arguments and configuration options should be taken from the current implementation of each script.

## Citation

If you use this code or the associated dataset in your research, please cite the accompanying paper:

> **AssemblyHAR: A smartwatch accelerometer dataset for activity recognition and event-log generation in manual assembly**

A version-specific DOI and citation details can be added here once the repository has been archived through Zenodo.

