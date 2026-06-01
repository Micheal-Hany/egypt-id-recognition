"""YOLO model loading and inference utilities."""

import os
from typing import Optional


def load_yolo_model(model_path: str):
    """
    Load a YOLO model from file.
    
    Args:
        model_path: Path to the .pt model file
        
    Returns:
        YOLO model object or None if load fails
    """
    if not os.path.exists(model_path):
        return None
    try:
        from ultralytics import YOLO
        return YOLO(model_path)
    except Exception as e:
        print(f"[WARN] Could not load model {model_path}: {e}")
        return None
