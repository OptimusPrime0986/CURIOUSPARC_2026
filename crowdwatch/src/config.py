"""Configuration management for CrowdWatch."""

from __future__ import annotations
import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

logger = logging.getLogger("crowdwatch.config")

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default.yaml"
DEFAULT_ZONES_PATH = Path(__file__).resolve().parent.parent / "config" / "zones.json"
DEFAULT_CALIB_PATH = Path(__file__).resolve().parent.parent / "config" / "calibration.json"


class ConfigManager:
    """Loads, validates, and provides unified access to configurations."""

    def __init__(
        self,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        zones_path: str | Path = DEFAULT_ZONES_PATH,
        calibration_path: str | Path = DEFAULT_CALIB_PATH,
    ) -> None:
        self.config_path = Path(config_path)
        self.zones_path = Path(zones_path)
        self.calibration_path = Path(calibration_path)

        self.config: Dict[str, Any] = {}
        self.zones_data: Dict[str, Any] = {}
        self.calibration_data: Dict[str, Any] = {}

        self.reload()

    def reload(self) -> None:
        """Reload all configuration files from disk."""
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f) or {}
            logger.info("Loaded system config from %s", self.config_path)
        else:
            logger.warning("Config path %s does not exist; using defaults", self.config_path)
            self.config = {}

        if self.zones_path.exists():
            with open(self.zones_path, "r", encoding="utf-8") as f:
                self.zones_data = json.load(f)
            logger.info("Loaded zones from %s", self.zones_path)
        else:
            logger.warning("Zones path %s does not exist", self.zones_path)
            self.zones_data = {"zones": [], "chokepoints": []}

        if self.calibration_path.exists():
            with open(self.calibration_path, "r", encoding="utf-8") as f:
                self.calibration_data = json.load(f)
            logger.info("Loaded calibration from %s", self.calibration_path)
        else:
            logger.warning("Calibration path %s does not exist", self.calibration_path)
            self.calibration_data = {}

    def get(self, section: str, key: Optional[str] = None, default: Any = None) -> Any:
        """Get a configuration value with fallback."""
        sec = self.config.get(section, {})
        if key is None:
            return sec
        if isinstance(sec, dict):
            return sec.get(key, default)
        return default

    def save_zones(self, zones_data: Dict[str, Any]) -> None:
        """Save zones and choke points to file."""
        self.zones_data = zones_data
        self.zones_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.zones_path, "w", encoding="utf-8") as f:
            json.dump(zones_data, f, indent=2)
        logger.info("Saved updated zones to %s", self.zones_path)

    def save_calibration(self, calibration_data: Dict[str, Any]) -> None:
        """Save calibration data to file."""
        self.calibration_data = calibration_data
        self.calibration_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.calibration_path, "w", encoding="utf-8") as f:
            json.dump(calibration_data, f, indent=2)
        logger.info("Saved updated calibration to %s", self.calibration_path)


# Global singleton instance
_GLOBAL_CONFIG: Optional[ConfigManager] = None


def get_config() -> ConfigManager:
    """Get the active configuration manager instance."""
    global _GLOBAL_CONFIG
    if _GLOBAL_CONFIG is None:
        _GLOBAL_CONFIG = ConfigManager()
    return _GLOBAL_CONFIG
