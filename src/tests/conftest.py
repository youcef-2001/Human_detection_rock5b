"""Test configuration and fixtures."""

import os
import sys
import pytest
import numpy as np
from pathlib import Path

# Add project root to path so ``src`` package is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.app.config import TestingConfig


@pytest.fixture
def config():
    """Provide testing configuration."""
    return TestingConfig()


@pytest.fixture
def sample_bgr_image():
    """Create sample BGR image (320x240)."""
    return np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)


@pytest.fixture
def sample_thermal_frame():
    """Create sample thermal frame (24x32 float32)."""
    # Generate realistic thermal data in range [5, 55]
    return np.random.uniform(5.0, 55.0, (24, 32)).astype(np.float32)


@pytest.fixture
def sample_gray_image():
    """Create sample grayscale image."""
    return np.random.randint(0, 256, (240, 320), dtype=np.uint8)
