import sys
from pathlib import Path

# Add parent directory to path so absolute imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset_generator.core import PreprocessingConfig, ActivityManager, WindowGenerator, SessionProcessor, DatasetBuilder
from dataset_generator.helpers import parse_session_folder_name


def main(preprocessing_config: PreprocessingConfig | None = None):
    """
    Main preprocessing pipeline orchestrator.
    
    This function:
    1. Initializes configuration and pipeline components
    2. Iterates over session folders and processes each session
    3. Aggregates results and builds final datasets
    """
    
    # Initialize configuration from config module unless provided by caller
    print("\n[1/4] Initializing configuration...")
    if preprocessing_config is None:
        preprocessing_config = PreprocessingConfig()
    print(f"  - Windowing mode: {'Full Activity' if preprocessing_config.full_activity_windowing else f'Sliding ({preprocessing_config.window_size_sec}s window, {preprocessing_config.step_size_sec}s step)'}")
    print(f"  - Activity grouping: {preprocessing_config.activity_grouping}")
    print(f"  - Motor separation: {preprocessing_config.separate_motors}")
    print(f"  - Reprocess all sessions: {getattr(preprocessing_config, 'reprocess_all_sessions', False)}")
    
    # Initialize pipeline components
    print("\n[2/4] Initializing pipeline components...")
    activity_manager = ActivityManager(preprocessing_config)
    window_generator = WindowGenerator(preprocessing_config)
    
    # Initialize dataset builder
    project_root = Path(__file__).resolve().parent.parent
    dataset_builder = DatasetBuilder(preprocessing_config, activity_manager, project_root)
    print(f"  - Dataset folders: {dataset_builder.dataset_root_features.name}, {dataset_builder.dataset_root_raw.name}")
    
    # Iterate through session folders
    print("\n[3/4] Processing sessions...")
    data_folder = project_root / "Data"
    processed_count = 0
    cached_count = 0
    failed_count = 0
    
    for folder in data_folder.iterdir():
        # Select only session folders inside the Data directory
        if not folder.is_dir() or folder.name == "Dataset" or not folder.name.startswith("session_"):
            continue
        
        # Extract session code and order condition from the session folder name
        print(f"\nSession: {folder.name}")
        try:
            session_code, order_condition = parse_session_folder_name(folder.name)
        except ValueError as e:
            print(f" Skipping: {e}")
            failed_count += 1
            continue
        
        # Create session processor
        session_processor = SessionProcessor(
            session_folder=folder,
            session_code=session_code,
            order_condition=order_condition,
            config=preprocessing_config,
            activity_manager=activity_manager,
            window_generator=window_generator
        )
        
        try:
            # Check if session needs reprocessing
            if session_processor.should_reprocess():
                if getattr(preprocessing_config, 'reprocess_all_sessions', False):
                    print("  Reprocess-all enabled; forcing reprocessing")
                elif preprocessing_config.activity_grouping:
                    print("  Activity grouping enabled; forcing reprocessing")
                elif preprocessing_config.full_activity_windowing:
                    print("  Full activity windowing enabled; forcing reprocessing")
                else:
                    print("  No cached results found; running preprocessing")
                
                # Process the session
                (features_df, raw_df, phase_stats_df, assembly_stats_df,
                activity_totals_df, activity_segments_df, acceleration_stats_df) = session_processor.process()
                processed_count += 1
                print(f"  Processed: {len(features_df)} feature windows, {len(raw_df)} raw windows")
            else:
                print("  Loading cached results")
                
                # Load cached results
                (features_df, raw_df, activity_totals_df, activity_segments_df,
                phase_stats_df, assembly_stats_df, acceleration_stats_df) = session_processor.load_cached_results()
                cached_count += 1
                print(f"  Loaded: {len(features_df)} feature windows, {len(raw_df)} raw windows")
            
            # Add session data to dataset builder
            dataset_builder.add_session_data(
                features_df=features_df,
                raw_df=raw_df,
                activity_totals_df=activity_totals_df,
                activity_segments_df=activity_segments_df,
                phase_stats_df=phase_stats_df,
                assembly_stats_df=assembly_stats_df,
                acceleration_stats_df=acceleration_stats_df,
                session_code=session_code
            )
            
        except Exception as e:
            print(f" Failed: {e}")
            failed_count += 1
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    print("\n" + "-"*80)
    print(f"Session processing complete:")
    print(f"  - Processed from scratch: {processed_count}")
    print(f"  - Loaded from cache: {cached_count}")
    print(f"  - Failed/Skipped: {failed_count}")
    print(f"  - Total collected: {len(dataset_builder.all_sessions_feat)} sessions")
    print("-"*80)
    
    # Build final datasets
    print("\n[4/4] Building final datasets...")
    dataset_builder.build_datasets()
    
    # Final summary
    print("\n" + "="*80)
    print("PREPROCESSING PIPELINE COMPLETE!")
    print("="*80)
    print(f"Datasets saved to: {dataset_builder.datasets_folder}")
    print(f"  - Features: {dataset_builder.dataset_root_features.name}")
    print(f"  - Raw data: {dataset_builder.dataset_root_raw.name}")
    if preprocessing_config.separate_motors:
        print(f"  - Motor-separated datasets:")
        print(f"      • {dataset_builder.dataset_root_features.name}_motor_1")
        print(f"      • {dataset_builder.dataset_root_features.name}_motor_2")
        print(f"      • {dataset_builder.dataset_root_raw.name}_motor_1")
        print(f"      • {dataset_builder.dataset_root_raw.name}_motor_2")
    print("="*80)


if __name__ == "__main__":
    main()
