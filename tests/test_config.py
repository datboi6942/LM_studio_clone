"""Tests for configuration module."""

import os
from pathlib import Path

import pytest

from app.config import Config


def test_config_defaults():
    """Test default configuration values."""
    assert Config.HOST == "127.0.0.1"
    assert Config.PORT == 8000
    assert Config.DEBUG is False
    assert Config.PUBLIC_MODE is False
    assert Config.LOG_LEVEL == "INFO"


def test_config_to_dict():
    """Test configuration export to dictionary."""
    config_dict = Config.to_dict()
    
    assert isinstance(config_dict, dict)
    assert "HOST" in config_dict
    assert "PORT" in config_dict
    assert "SECRET_KEY" in config_dict


def test_config_validate_creates_directories(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test that validate creates required directories."""
    models_dir = tmp_path / "models"
    chroma_dir = tmp_path / "chroma"
    
    monkeypatch.setattr(Config, "MODELS_DIR", models_dir)
    monkeypatch.setattr(Config, "CHROMA_PERSIST_DIR", chroma_dir)
    
    assert not models_dir.exists()
    assert not chroma_dir.exists()
    
    Config.validate()
    
    assert models_dir.exists()
    assert chroma_dir.exists()


def test_config_validate_public_mode_security(monkeypatch: pytest.MonkeyPatch):
    """Test that validate raises error when PUBLIC_MODE enabled with default secret."""
    monkeypatch.setattr(Config, "PUBLIC_MODE", True)
    monkeypatch.setattr(Config, "SECRET_KEY", "dev-secret-key-change-in-production")
    
    with pytest.raises(ValueError, match="SECRET_KEY must be changed"):
        Config.validate() 