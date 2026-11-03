"""multiview3d — 3D-координаты и траектории движущихся объектов по нескольким
калиброванным камерам: детекция, сопоставление видов, триангуляция, Калман,
траектории, метрики, визуализация.

Быстрый старт: ``sim.make_sim_setup`` -> ``pipeline.PipelineConfig.from_json``
-> ``MultiViewPipeline.run_offline`` -> ``metrics.evaluate``.
"""

from .types import (
    CameraCalibration,
    Detection2D,
    Observation3D,
    Track,
    TrackPoint,
)

__version__ = "0.1.0"
__all__ = [
    "CameraCalibration",
    "Detection2D",
    "Observation3D",
    "Track",
    "TrackPoint",
    "__version__",
]
