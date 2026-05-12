"""
MHARS — Stage 4: Sensor Acquisition Module
=============================================
Hardware abstraction layer for acquiring audio and visual
sensor data in deployed environments.

Provides three acquisition modes:
  1. LIVE     — real microphone (PyAudio) + real camera (OpenCV)
  2. FILE     — pre-recorded .wav audio + saved thermal images
  3. SIMULATE — synthesised signals from temperature dynamics
                (default for demo/research, no hardware needed)

Usage:
    from stage4_hardware.sensor_acquisition import SensorAcquisition
    acq = SensorAcquisition(mode="simulate")
    audio_features = acq.acquire_audio()
    image_features = acq.acquire_image(current_temp=72.0)
"""

import os
import numpy as np
from typing import Dict, Any, Optional


class SensorAcquisition:
    """
    Unified hardware abstraction for multi-modal sensor input.

    Supports three modes:
      - "live"     : Real hardware (mic + camera)
      - "file"     : Pre-recorded files
      - "simulate" : Synthesised from thermal dynamics (no HW needed)
    """

    def __init__(self, mode: str = "simulate", verbose: bool = False):
        assert mode in ("live", "file", "simulate"), f"Invalid mode: {mode}"
        self.mode = mode
        self.verbose = verbose

        self._audio_device = None
        self._camera_device = None

        if mode == "live":
            self._init_live_audio()
            self._init_live_camera()
        elif mode == "file":
            self._audio_dir = os.environ.get("MHARS_AUDIO_DIR", "data/audio")
            self._image_dir = os.environ.get("MHARS_IMAGE_DIR", "data/thermal")
            self._file_index = 0

    # ── Audio acquisition ──────────────────────────────────────────────────────
    def _init_live_audio(self):
        """Initialise microphone via PyAudio (optional dependency)."""
        try:
            import pyaudio
            self._pa = pyaudio.PyAudio()
            # Find default input device
            info = self._pa.get_default_input_device_info()
            self._audio_device = {
                "name": info["name"],
                "rate": int(info["defaultSampleRate"]),
                "channels": 1,
                "chunk_size": 1024,
                "format": pyaudio.paFloat32,
            }
            if self.verbose:
                print(f"  [AUDIO] Microphone: {info['name']} @ {info['defaultSampleRate']}Hz")
        except ImportError:
            print("  ⚠  PyAudio not installed — audio acquisition disabled")
            print("     Install with: pip install pyaudio")
            self.mode = "simulate"
        except Exception as e:
            print(f"  ⚠  Audio init failed: {e} — falling back to simulate")
            self.mode = "simulate"

    def _init_live_camera(self):
        """Initialise camera via OpenCV (optional dependency)."""
        try:
            import cv2
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                self._camera_device = cap
                if self.verbose:
                    print(f"  [CAMERA] Camera opened (device 0)")
            else:
                print("  ⚠  No camera found — falling back to simulate")
                self.mode = "simulate"
        except ImportError:
            print("  ⚠  OpenCV not installed — camera acquisition disabled")
            print("     Install with: pip install opencv-python")
            self.mode = "simulate"
        except Exception as e:
            print(f"  ⚠  Camera init failed: {e} — falling back to simulate")
            self.mode = "simulate"

    def acquire_audio(self, duration_sec: float = 1.0,
                      current_temp: float = 50.0,
                      safe_max: float = 80.0) -> Dict[str, Any]:
        """
        Acquire audio features from the configured source.

        Returns dict with:
          - audio_score: float 0–1 (anomaly severity)
          - audio_variance: float (confidence metric)
          - source: str ("mic", "file", or "simulated")
        """
        if self.mode == "live":
            return self._acquire_live_audio(duration_sec)
        elif self.mode == "file":
            return self._acquire_file_audio()
        else:
            return self._simulate_audio(current_temp, safe_max)

    def acquire_image(self, current_temp: float = 50.0,
                      safe_max: float = 80.0) -> Dict[str, Any]:
        """
        Acquire thermal image features from the configured source.

        Returns dict with:
          - cnn_score: float 0–1 (hotspot severity)
          - grid_variance: float (spatial heat distribution)
          - source: str ("camera", "file", or "simulated")
        """
        if self.mode == "live":
            return self._acquire_live_image()
        elif self.mode == "file":
            return self._acquire_file_image()
        else:
            return self._simulate_image(current_temp, safe_max)

    # ── Live hardware implementations ──────────────────────────────────────────
    def _acquire_live_audio(self, duration_sec: float) -> Dict[str, Any]:
        """Record audio from microphone and extract MFCC features."""
        import pyaudio
        dev = self._audio_device
        stream = self._pa.open(
            format=dev["format"],
            channels=dev["channels"],
            rate=dev["rate"],
            input=True,
            frames_per_buffer=dev["chunk_size"],
        )
        frames = []
        n_chunks = int(dev["rate"] * duration_sec / dev["chunk_size"])
        for _ in range(n_chunks):
            data = stream.read(dev["chunk_size"], exception_on_overflow=False)
            frames.append(np.frombuffer(data, dtype=np.float32))
        stream.stop_stream()
        stream.close()

        audio_signal = np.concatenate(frames)

        # Extract MFCC features
        try:
            from stage2_ml.audio_mfcc import AudioPipeline
            pipeline = AudioPipeline()
            result = pipeline.process_signal(audio_signal, dev["rate"])
            return {**result, "source": "mic"}
        except Exception:
            rms = float(np.sqrt(np.mean(audio_signal ** 2)))
            return {"audio_score": min(rms * 10, 1.0), "audio_variance": 0.1, "source": "mic"}

    def _acquire_live_image(self) -> Dict[str, Any]:
        """Capture frame from camera and run CNN hotspot detection."""
        import cv2
        ret, frame = self._camera_device.read()
        if not ret:
            return {"cnn_score": 0.5, "grid_variance": 0.1, "source": "camera_fail"}

        # Convert to thermal-like grayscale (real deployment uses FLIR camera)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        try:
            from stage2_ml.mobilenet_cnn import ThermalHotspotDetector
            detector = ThermalHotspotDetector()
            result = detector.predict_from_array(gray)
            return {**result, "source": "camera"}
        except Exception:
            intensity = float(np.mean(gray)) / 255.0
            return {"cnn_score": intensity, "grid_variance": float(np.std(gray / 255.0)), "source": "camera"}

    # ── File-based implementations ─────────────────────────────────────────────
    def _acquire_file_audio(self) -> Dict[str, Any]:
        """Load audio from pre-recorded WAV file."""
        try:
            import wave
            files = sorted([f for f in os.listdir(self._audio_dir) if f.endswith(".wav")])
            if not files:
                return self._simulate_audio(50.0, 80.0)

            filepath = os.path.join(self._audio_dir, files[self._file_index % len(files)])
            self._file_index += 1

            with wave.open(filepath, 'rb') as wf:
                rate = wf.getframerate()
                frames = wf.readframes(wf.getnframes())
                signal = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

            from stage2_ml.audio_mfcc import AudioPipeline
            pipeline = AudioPipeline()
            result = pipeline.process_signal(signal, rate)
            return {**result, "source": "file"}
        except Exception as e:
            if self.verbose:
                print(f"  [AUDIO] File load failed: {e}")
            return self._simulate_audio(50.0, 80.0)

    def _acquire_file_image(self) -> Dict[str, Any]:
        """Load thermal image from file."""
        try:
            from PIL import Image
            files = sorted([f for f in os.listdir(self._image_dir)
                           if f.endswith((".png", ".jpg", ".bmp"))])
            if not files:
                return self._simulate_image(50.0, 80.0)

            filepath = os.path.join(self._image_dir, files[self._file_index % len(files)])

            from stage2_ml.mobilenet_cnn import ThermalHotspotDetector
            detector = ThermalHotspotDetector()
            result = detector.predict(filepath)
            return {**result, "source": "file"}
        except Exception as e:
            if self.verbose:
                print(f"  [IMAGE] File load failed: {e}")
            return self._simulate_image(50.0, 80.0)

    # ── Simulation implementations (no hardware needed) ────────────────────────
    def _simulate_audio(self, current_temp: float, safe_max: float) -> Dict[str, Any]:
        """Synthesise audio anomaly score from thermal dynamics."""
        try:
            from stage2_ml.audio_mfcc import AudioPipeline
            pipeline = AudioPipeline()
            result = pipeline.process_from_temperature(current_temp, safe_max)
            return {**result, "source": "simulated"}
        except Exception:
            ratio = max(0, (current_temp - safe_max * 0.6) / (safe_max * 0.4))
            return {
                "audio_score": float(np.clip(ratio + np.random.normal(0, 0.05), 0, 1)),
                "audio_variance": 0.08,
                "source": "simulated",
            }

    def _simulate_image(self, current_temp: float, safe_max: float) -> Dict[str, Any]:
        """Synthesise CNN hotspot score from thermal dynamics."""
        try:
            from stage2_ml.mobilenet_cnn import ThermalHotspotDetector
            detector = ThermalHotspotDetector()
            result = detector.predict_from_temperature(current_temp, safe_max)
            return {**result, "source": "simulated"}
        except Exception:
            ratio = max(0, (current_temp - safe_max * 0.5) / (safe_max * 0.5))
            return {
                "cnn_score": float(np.clip(ratio + np.random.normal(0, 0.03), 0, 1)),
                "grid_variance": 0.05,
                "source": "simulated",
            }

    # ── Cleanup ────────────────────────────────────────────────────────────────
    def close(self):
        """Release hardware resources."""
        if self._camera_device is not None:
            self._camera_device.release()
            self._camera_device = None
        if hasattr(self, '_pa') and self._pa is not None:
            self._pa.terminate()
            self._pa = None

    def __del__(self):
        self.close()
