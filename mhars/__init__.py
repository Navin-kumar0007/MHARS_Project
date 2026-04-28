"""
MHARS — Multi-modal Hybrid Adaptive Response System
=====================================================
Universal thermal management for IoT environments.

Quick start:
    from mhars import MHARS
    system = MHARS(machine_type_id=0)   # 0=CPU 1=Motor 2=Server 3=Engine
    result = system.run(temp_celsius=72.5)
    print(result.action)
    print(result.alert)

With real LLM:
    system = MHARS(machine_type_id=0,
                   llm_path="models/Phi-3-mini-4k-instruct-q4.gguf")

Live dashboard:
    from mhars import MHARS, Dashboard
    system = MHARS(machine_type_id=0)
    dash   = Dashboard(system)
    dash.start(source="simulation")

Train all models:
    from mhars import MHARSTrainer
    trainer = MHARSTrainer()
    trainer.train_all()
"""

from mhars.core      import MHARS, MHARSResult
from mhars.config    import Config
from mhars.trainer   import MHARSTrainer
from mhars.dashboard import Dashboard
from mhars.models    import ThermalLSTM, ThermalAutoencoder

__version__ = "1.0.0"
__author__  = "MHARS Research Project"

__all__ = [
    "MHARS", "MHARSResult", "Config",
    "MHARSTrainer", "Dashboard",
    "ThermalLSTM", "ThermalAutoencoder",
]