"""
MHARS — Stage 4: Hardware Sensor Drivers
==========================================
Interfaces with real I2C sensors (MLX90640, MPU6050).
Falls back to simulated data if no I2C bus is detected.
"""

import time
import numpy as np

try:
    import smbus2
    import board
    import busio
    import adafruit_mlx90640
    HARDWARE_AVAILABLE = True
except ImportError:
    HARDWARE_AVAILABLE = False

class MLX90640_Camera:
    def __init__(self, simulate: bool = False):
        self.simulate = simulate or not HARDWARE_AVAILABLE
        self.mlx = None
        if not self.simulate:
            try:
                i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)
                self.mlx = adafruit_mlx90640.MLX90640(i2c)
                self.mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_2_HZ
                print("[Hardware] MLX90640 Thermal Camera connected.")
            except Exception as e:
                print(f"[Hardware] MLX90640 connection failed: {e}. Falling back to simulation.")
                self.simulate = True

    def get_frame(self) -> np.ndarray:
        if self.simulate:
            # Return a synthetic 24x32 thermal frame
            frame = np.random.normal(30.0, 2.0, (24, 32))
            return frame.astype(np.float32)
        else:
            frame = [0] * 768
            try:
                self.mlx.getFrame(frame)
                return np.array(frame).reshape((24, 32)).astype(np.float32)
            except ValueError:
                # Occasional read errors on I2C
                return np.random.normal(30.0, 2.0, (24, 32)).astype(np.float32)

class MPU6050_Vibration:
    def __init__(self, simulate: bool = False):
        self.simulate = simulate or not HARDWARE_AVAILABLE
        self.bus = None
        if not self.simulate:
            try:
                self.bus = smbus2.SMBus(1)
                self.address = 0x68
                self.bus.write_byte_data(self.address, 0x6B, 0) # Wake up
                print("[Hardware] MPU6050 Accelerometer connected.")
            except Exception as e:
                print(f"[Hardware] MPU6050 connection failed: {e}. Falling back to simulation.")
                self.simulate = True

    def get_vibration(self) -> dict:
        if self.simulate:
            return {
                "accel_x": np.random.normal(0, 0.1),
                "accel_y": np.random.normal(0, 0.1),
                "accel_z": np.random.normal(9.8, 0.1)
            }
        else:
            try:
                # Read 6 bytes of accel data starting at 0x3B
                data = self.bus.read_i2c_block_data(self.address, 0x3B, 6)
                x = self._convert(data[0], data[1])
                y = self._convert(data[2], data[3])
                z = self._convert(data[4], data[5])
                return {"accel_x": x, "accel_y": y, "accel_z": z}
            except Exception:
                return {"accel_x": 0.0, "accel_y": 0.0, "accel_z": 9.8}

    def _convert(self, high, low):
        val = (high << 8) | low
        if val > 32767:
            val -= 65536
        return val / 16384.0 * 9.8 # Convert to m/s^2 approx
