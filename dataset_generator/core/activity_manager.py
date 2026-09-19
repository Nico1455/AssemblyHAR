from __future__ import annotations
from typing import List, Tuple
import pandas as pd

from ..helpers import MOTOR1_ACTIVITIES, MOTOR2_ACTIVITIES
from .signal_preprocessing_config import PreprocessingConfig


class ActivityManager:
    """
    Manages activity labels, grouping, and motor-based operations.
    
    This class encapsulates all logic related to activity management including:
    - Motor activity definitions and grouping
    - Activity label transformations
    - Motor-based dataset filtering
    
    The ActivityManager works in conjunction with PreprocessingConfig to
    determine whether activity grouping is enabled and what the grouping
    rules are. When grouping is enabled, it remaps activity labels to
    group names (e.g., mapping multiple assembly activities to "Assembly").
    
    Attributes:
        config (PreprocessingConfig): Configuration instance controlling
            grouping behavior.
        motor1_activities (list[str]): Activities performed by motor 1.
            Either the original list or grouped list depending on config.
        motor2_activities (list[str]): Activities performed by motor 2.
            Either the original list or grouped list depending on config.
    """
    
    def __init__(self, config: PreprocessingConfig):
        """
        Initialize the ActivityManager.
        
        If activity grouping is enabled in the configuration, applies the
        grouping rules to the motor activity lists immediately.
        
        Parameters:
            config (PreprocessingConfig): Configuration instance specifying
                activity grouping settings.
        
        Example:
            >>> config = PreprocessingConfig(activity_grouping=False)
            >>> manager = ActivityManager(config)
            >>> len(manager.motor1_activities) > 10  # Full list
            True
        """
        self.config = config
        self.motor1_activities = MOTOR1_ACTIVITIES.copy()
        self.motor2_activities = MOTOR2_ACTIVITIES.copy()
        
        # Apply grouping if enabled
        if config.activity_grouping and config.activity_groups:
            self._apply_grouping()
    
    def _apply_grouping(self):
        """
        Apply activity grouping to motor activity lists.
        
        This internal method remaps the motor activity lists to group names
        based on the activity_groups configuration. Each original activity
        is replaced with its group name (if defined in activity_groups).
        
        Activities that are not in any group are preserved unchanged.
        
        For example, if activity_groups = {"Assembly": ["Pick up motor housing", "Insert rotor"]},
        then both "Pick up motor housing" and "Insert rotor" would be mapped
        to "Assembly" in the activity lists, while any activities not in a group
        remain as original activity names.
        
        This is called automatically during __init__ if grouping is enabled.
        """
        grouped_motor1 = []
        grouped_motor2 = []
        
        # Build a reverse mapping: activity -> group_name
        activity_to_group = {}
        for group_name, original_activities in self.config.activity_groups.items():
            for activity in original_activities:
                activity_to_group[activity] = group_name
        
        # Process motor1 activities: grouped activities become group names, ungrouped stay as-is
        for activity in MOTOR1_ACTIVITIES:
            if activity in activity_to_group:
                group_name = activity_to_group[activity]
                if group_name not in grouped_motor1:
                    grouped_motor1.append(group_name)
            else:
                # Activity not in any group, keep it unchanged
                if activity not in grouped_motor1:
                    grouped_motor1.append(activity)
        
        # Process motor2 activities: grouped activities become group names, ungrouped stay as-is
        for activity in MOTOR2_ACTIVITIES:
            if activity in activity_to_group:
                group_name = activity_to_group[activity]
                if group_name not in grouped_motor2:
                    grouped_motor2.append(group_name)
            else:
                # Activity not in any group, keep it unchanged
                if activity not in grouped_motor2:
                    grouped_motor2.append(activity)
        
        # Update motor activity lists
        self.motor1_activities = grouped_motor1
        self.motor2_activities = grouped_motor2
    
    def apply_activity_grouping(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply activity grouping to a dataframe with a 'label' column.
        
        Remaps all activity labels in the dataframe to their corresponding
        group names according to the activity_groups configuration.
        
        Only performs remapping if activity_grouping is enabled in config.
        If grouping is disabled, returns the dataframe unchanged.
        
        Parameters:
            df (pd.DataFrame): Input dataframe with 'label' column containing
                original activity names.
        
        Returns:
            pd.DataFrame: New dataframe with grouped labels.
                The original dataframe is not modified (copy is made).
        
        Example:
            >>> config = PreprocessingConfig(
            ...     activity_grouping=True,
            ...     activity_groups={"Assembly": ["Pick up motor housing"]}
            ... )
            >>> manager = ActivityManager(config)
            >>> df = pd.DataFrame({'label': ['Pick up motor housing', 'Insert rotor']})
            >>> df_grouped = manager.apply_activity_grouping(df)
            >>> df_grouped['label'].tolist()
            ['Assembly', 'Insert rotor']
        """
        # Return unchanged if grouping is disabled
        if not self.config.activity_grouping or not self.config.activity_groups:
            return df
        
        df = df.copy()
        
        # Create reverse mapping: original_activity -> group_name
        activity_to_group = {}
        for group_name, activities in self.config.activity_groups.items():
            for activity in activities:
                activity_to_group[activity] = group_name
        
        # Apply mapping to label column
        # Uses map() which returns NaN for unmapped values, then fills them with original
        df['label'] = df['label'].map(lambda x: activity_to_group.get(x, x))
        
        return df
    
    def filter_by_motor(self, df: pd.DataFrame, motor: str) -> pd.DataFrame:
        """
        Filter a dataframe to only include activities from a specific motor.
        
        Removes all rows whose 'label' column contains an activity not
        performed by the specified motor. Uses the motor activity lists
        (which may be grouped) to determine which activities to keep.
        
        Parameters:
            df (pd.DataFrame): Input dataframe with 'label' column.
            motor (str): Motor to filter by. Either "motor1" or "motor2".
        
        Returns:
            pd.DataFrame: Filtered dataframe containing only rows with activities
                from the specified motor. Returns a copy of the dataframe.
        
        Raises:
            ValueError: If motor is not "motor1" or "motor2".
        
        Example:
            >>> config = PreprocessingConfig()
            >>> manager = ActivityManager(config)
            >>> df = pd.DataFrame({
            ...     'label': ['Pick up motor housing', 'Pick up rotor'],
            ...     'value': [1, 2]
            ... })
            >>> df_m1 = manager.filter_by_motor(df, "motor1")
            >>> len(df_m1)
            1  # Only motor1 activities included
        """
        # Determine which activities belong to the requested motor
        if motor == "motor1":
            activities = self.motor1_activities
        elif motor == "motor2":
            activities = self.motor2_activities
        else:
            raise ValueError(
                f"Invalid motor: {motor}. Must be 'motor1' or 'motor2'"
            )
        
        # Filter dataframe to only include rows with motor's activities
        return df[df['label'].isin(activities)].copy()
    
    def get_motor_activities(self) -> Tuple[List[str], List[str]]:
        """
        Get the current motor activity lists (potentially grouped).
        
        Returns the activity lists as they currently exist, which may be
        the original lists or grouped lists depending on configuration
        and whether _apply_grouping() was called.
        
        These lists are used throughout the pipeline for filtering, validation,
        and documentation purposes.
        
        Returns:
            tuple[list[str], list[str]]: (motor1_activities, motor2_activities)
                where each list contains activity names for that motor.
        
        Example:
            >>> config = PreprocessingConfig(activity_grouping=False)
            >>> manager = ActivityManager(config)
            >>> m1, m2 = manager.get_motor_activities()
            >>> len(m1)
            9  # Full original activity list for motor1
            >>> len(m2)
            10  # Full original activity list for motor2
        """
        return self.motor1_activities, self.motor2_activities
    
    def assign_motor_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign motor IDs to a dataframe based on activity labels.
        
        Creates a new 'motor_id' column (1 or 2) based on which motor
        performs each activity in the 'label' column. Uses the current
        motor activity lists (which may be grouped).
        
        Activities not found in either motor's activity list are assigned
        motor_id = 0 (unknown motor).
        
        Parameters:
            df (pd.DataFrame): Input dataframe with 'label' column containing
                activity names.
        
        Returns:
            pd.DataFrame: Copy of input dataframe with new 'motor_id' column.
                Values are 1 (motor 1), 2 (motor 2), or 0 (unknown).
        """
        df = df.copy()
        
        # Initialize motor_id column with 0 (unknown)
        df['motor_id'] = 0
        
        # Assign motor 1
        df.loc[df['label'].isin(self.motor1_activities), 'motor_id'] = 1
        
        # Assign motor 2
        df.loc[df['label'].isin(self.motor2_activities), 'motor_id'] = 2
        
        return df
