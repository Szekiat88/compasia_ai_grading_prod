# core/models.py
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from .enums import Grade, DefectType, DeviceArea, ScratchType


@dataclass
class ScratchDefect:
    """Represents a scratch defect with its type and size"""
    scratch_type: ScratchType
    size_mm: float
    area: DeviceArea


@dataclass
class AreaDefects:
    """Tracks defects within a specific area"""
    area: DeviceArea
    micro_scratches: List[float] = field(default_factory=list)  # sizes in mm
    minor_scratches: List[float] = field(default_factory=list)  # sizes in mm
    major_scratches: List[float] = field(default_factory=list)  # sizes in mm


@dataclass
class DefectCount:
    """Enhanced defect count now supporting area-based tracking"""
    # Legacy counts (for backward compatibility)
    minor_scratches: int = 0
    major_scratches: int = 0
    dents: List[float] = field(default_factory=list)  # Will store dent sizes in mm
    discoloration: List[float] = field(default_factory=list)  # Will store sizes in mm
    cover_spots: List[float] = field(default_factory=list)  # Will store sizes in mm

    # New area-based tracking
    area_defects: Dict[DeviceArea, AreaDefects] = field(default_factory=dict)

    def __post_init__(self):
        self.dents = self.dents or []
        self.discoloration = self.discoloration or []
        self.cover_spots = self.cover_spots or []
        self.area_defects = self.area_defects or {}


@dataclass
class GradingResult:
    grade: Grade
    defect_count: DefectCount
    details: str