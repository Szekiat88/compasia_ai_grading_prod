# grading_system/utils/device_dimensions.py
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple


class DeviceDimensions:
    """Handles device dimension lookups and calculations"""

    def __init__(self, csv_path: Optional[str] = None):
        """
        Initialize with path to dimensions CSV file

        Args:
            csv_path: Path to device_dimensions.csv. If None, uses default path
        """
        if csv_path is None:
            # Default to looking in a 'data' directory at project root
            csv_path = Path(__file__).parent.parent.parent / 'grading_system'/ 'data' / 'device_dimensions.csv'

        self.dimensions_df = self._load_dimensions(csv_path)

    def _load_dimensions(self, csv_path: Path) -> pd.DataFrame:
        """Load and validate device dimensions CSV"""
        try:
            df = pd.read_csv(csv_path)
            required_columns = ['Brand', 'Model', 'Height (mm)', 'Width (mm)', 'Thickness (mm)']
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                raise ValueError(f"Missing required columns in CSV: {missing_columns}")

            return df
        except Exception as e:
            print(f"Error loading device dimensions: {str(e)}")
            print("Using default dimensions")
            # Create a minimal DataFrame with default values
            return pd.DataFrame({
                'Brand': ['Default'],
                'Model': ['Default'],
                'Height (mm)': [140],
                'Width (mm)': [70],
                'Thickness (mm)': [7.5]
            })

    def get_dimensions(self, model: Optional[str] = None) -> Tuple[float, float, float]:
        """
        Get dimensions for specified model or average dimensions

        Args:
            model: Device model name

        Returns:
            Tuple of (height_mm, width_mm, thickness_mm)
        """
        if model and model in self.dimensions_df['Model'].values:
            row = self.dimensions_df[self.dimensions_df['Model'] == model].iloc[0]
        else:
            # Use average dimensions if model not specified or not found
            row = self.dimensions_df.select_dtypes(include=['float64', 'int64']).mean()

        return (
            row['Height (mm)'],
            row['Width (mm)'],
            row['Thickness (mm)']
        )