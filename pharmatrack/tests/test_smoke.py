"""
PharmaTrack Smoke Tests
Basic tests to verify project structure and imports
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_config_import():
    """Test that config module can be imported."""
    from pharmatrack import config
    assert config.PROJECT_ROOT is not None
    assert config.EXPIRY_WARNING_THRESHOLD == 30
    assert config.EXPIRY_CRITICAL_THRESHOLD == 7


def test_config_directories():
    """Test that config creates directories."""
    from pharmatrack import config
    assert config.DATA_DIR.exists()
    assert config.REPORTS_DIR.exists()


def test_required_libraries():
    """Test that required libraries can be imported."""
    import pandas
    import numpy
    import matplotlib
    import seaborn
    import streamlit
    import pytest
    assert pandas.__version__ is not None
    assert numpy.__version__ is not None


def test_project_structure():
    """Test that required directories exist."""
    pharmatrack_dir = Path(__file__).parent.parent
    
    # Check main structure
    assert (pharmatrack_dir / "app.py").exists()
    assert (pharmatrack_dir / "config.py").exists()
    assert (pharmatrack_dir / "requirements.txt").exists()
    
    # Check source directories
    assert (pharmatrack_dir / "src" / "data").exists()
    assert (pharmatrack_dir / "src" / "inventory").exists()
    assert (pharmatrack_dir / "src" / "analytics").exists()
    assert (pharmatrack_dir / "src" / "expiry").exists()
    assert (pharmatrack_dir / "src" / "visualization").exists()
    
    # Check other directories
    assert (pharmatrack_dir / "pages").exists()
    assert (pharmatrack_dir / "tests").exists()
    assert (pharmatrack_dir / "reports").exists()
    assert (pharmatrack_dir / "data").exists()


def test_src_packages_import():
    """Test that src packages can be imported."""
    from pharmatrack.src import data, inventory, analytics, expiry, visualization
    assert data is not None
    assert inventory is not None
    assert analytics is not None
    assert expiry is not None
    assert visualization is not None
