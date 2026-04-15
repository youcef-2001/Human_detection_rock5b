#!/usr/bin/env python
"""Quick start and example usage of the Human Detection API."""

import json
import requests
import numpy as np
from pathlib import Path
from io import BytesIO


class APIClient:
    """Simple client for Human Detection API."""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        """Initialize API client."""
        self.base_url = base_url
    
    def health_check(self) -> dict:
        """Check API health status."""
        response = requests.get(f"{self.base_url}/health")
        return response.json()
    
    def hello(self) -> dict:
        """Get hello world greeting."""
        response = requests.get(f"{self.base_url}/hello/")
        return response.json()
    
    def detect_image(self, image_path: str) -> dict:
        """
        Detect humans and objects in image file.
        
        Args:
            image_path: Path to image file (JPEG, PNG).
        
        Returns:
            Detection results with human_count and hot_object_count.
        """
        with open(image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post(
                f"{self.base_url}/inference/detect",
                files=files
            )
        return response.json()
    
    def detect_thermal(self, thermal_data: np.ndarray) -> dict:
        """
        Detect humans and objects in thermal frame.
        
        Args:
            thermal_data: Thermal frame (24x32 float32) or raw bytes.
        
        Returns:
            Detection results.
        """
        if isinstance(thermal_data, np.ndarray):
            thermal_bytes = thermal_data.astype(np.float32).tobytes()
        else:
            thermal_bytes = thermal_data
        
        files = {'image': ('thermal.bin', BytesIO(thermal_bytes))}
        response = requests.post(
            f"{self.base_url}/inference/detect",
            files=files
        )
        return response.json()


def example_basic_usage():
    """Example: Basic API usage."""
    print("=" * 50)
    print("EXAMPLE 1: Basic API Usage")
    print("=" * 50)
    
    client = APIClient()
    
    # Health check
    print("\n1. Health Check:")
    health = client.health_check()
    print(f"   Status: {health}")
    
    # Hello endpoint
    print("\n2. Hello World:")
    greeting = client.hello()
    print(f"   Message: {greeting['message']}")


def example_thermal_detection():
    """Example: Thermal frame detection."""
    print("\n" + "=" * 50)
    print("EXAMPLE 2: Thermal Frame Detection")
    print("=" * 50)
    
    client = APIClient()
    
    # Generate synthetic thermal frame
    thermal_frame = np.random.uniform(5, 55, (24, 32)).astype(np.float32)
    
    print(f"\nThermal frame: {thermal_frame.shape}")
    print(f"Temperature range: [{thermal_frame.min():.1f}, {thermal_frame.max():.1f}]°C")
    
    # Detect
    print("\nSending to API...")
    result = client.detect_thermal(thermal_frame)
    print(f"\nResults:")
    print(f"  - Humans detected: {result['human_count']}")
    print(f"  - Hot objects detected: {result['hot_object_count']}")
    print(f"  - Success: {result.get('success', 'N/A')}")


def example_image_detection():
    """Example: Standard image detection."""
    print("\n" + "=" * 50)
    print("EXAMPLE 3: Standard Image Detection")
    print("=" * 50)
    
    # Note: Image file must exist
    sample_image = Path("sample.jpg")
    
    if not sample_image.exists():
        print(f"\n⚠️  Sample image not found: {sample_image}")
        print("   To test, provide an image file named 'sample.jpg'")
        
        # Create a dummy image for demonstration
        import cv2
        img = np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)
        cv2.imwrite("sample.jpg", img)
        print("   Created dummy image: sample.jpg")
    
    client = APIClient()
    
    print(f"\nDetecting in: {sample_image}")
    result = client.detect_image(str(sample_image))
    
    print(f"\nResults:")
    print(f"  - Humans detected: {result.get('human_count', 'Error')}")
    print(f"  - Hot objects detected: {result.get('hot_object_count', 'Error')}")


def example_batch_processing():
    """Example: Batch processing multiple frames."""
    print("\n" + "=" * 50)
    print("EXAMPLE 4: Batch Processing")
    print("=" * 50)
    
    client = APIClient()
    
    # Process 5 synthetic thermal frames
    print("\nProcessing 5 thermal frames...\n")
    
    total_humans = 0
    total_objects = 0
    
    for i in range(5):
        # Generate thermal frame
        thermal_frame = np.random.uniform(5, 55, (24, 32)).astype(np.float32)
        
        # Detect
        result = client.detect_thermal(thermal_frame)
        
        humans = result.get('human_count', 0)
        objects = result.get('hot_object_count', 0)
        
        total_humans += humans
        total_objects += objects
        
        print(f"Frame {i+1}: {humans} humans, {objects} objects")
    
    print(f"\nTotal processed: 5 frames")
    print(f"Total humans detected: {total_humans}")
    print(f"Total objects detected: {total_objects}")


def example_curl_commands():
    """Show equivalent curl commands."""
    print("\n" + "=" * 50)
    print("EQUIVALENT CURL COMMANDS")
    print("=" * 50)
    
    print("""
# Health check
curl http://localhost:5000/health

# Hello world
curl http://localhost:5000/hello/

# Detect in image
curl -X POST -F "image=@image.jpg" \\
     http://localhost:5000/inference/detect

# Detect in thermal data
curl -X POST -F "image=@thermal.bin" \\
     http://localhost:5000/inference/detect
    """)


def main():
    """Run all examples."""
    print("\n" + "=" * 50)
    print("HUMAN DETECTION API - USAGE EXAMPLES")
    print("=" * 50)
    print("\nMake sure the server is running:")
    print("  python run.py")
    print("\nPress Enter to try the examples...\n")
    
    try:
        example_basic_usage()
        example_thermal_detection()
        example_image_detection()
        example_batch_processing()
        example_curl_commands()
        
        print("\n" + "=" * 50)
        print("✓ All examples completed!")
        print("=" * 50 + "\n")
    
    except requests.ConnectionError:
        print("\n❌ ERROR: Cannot connect to API!")
        print("   Make sure the server is running: python run.py")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
