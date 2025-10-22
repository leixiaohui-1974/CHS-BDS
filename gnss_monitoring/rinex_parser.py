"""
RINEX文件解析器模块

本模块提供标准RINEX格式文件的解析功能：
- RINEX观测文件（OBS）v2.x和v3.x
- RINEX导航文件（NAV）
- SNR（信噪比）数据提取
- 卫星高度角、方位角计算
- 支持GPS、GLONASS、Galileo、BeiDou等多系统

RINEX格式参考：
- RINEX 2.11: ftp://igs.org/pub/data/format/rinex211.txt
- RINEX 3.04: ftp://igs.org/pub/data/format/rinex304.pdf

Author: Lei Xiaohui
Date: 2025-01-22
"""

import os
import re
import gzip
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np

from .logger import get_logger
from .exceptions import CHSBDSException

logger = get_logger(__name__)


class RINEXError(CHSBDSException):
    """RINEX解析异常"""
    pass


@dataclass
class RINEXHeader:
    """RINEX文件头信息"""
    version: float
    file_type: str  # 'O' for OBS, 'N' for NAV
    satellite_system: str  # 'G' GPS, 'R' GLONASS, 'E' Galileo, 'C' BeiDou
    marker_name: str = ""
    receiver_type: str = ""
    antenna_type: str = ""
    approx_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    antenna_delta: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    time_of_first_obs: Optional[datetime] = None
    time_of_last_obs: Optional[datetime] = None
    interval: float = 1.0
    leap_seconds: int = 18
    obs_types: List[str] = None

    def __post_init__(self):
        if self.obs_types is None:
            self.obs_types = []


@dataclass
class Observation:
    """GNSS观测数据"""
    epoch: datetime
    satellite: str  # 例如: "G01", "C05", "E12"
    observations: Dict[str, float]  # 观测类型 -> 观测值
    lli: Dict[str, int] = None  # Loss of Lock Indicator
    signal_strength: Dict[str, int] = None

    def __post_init__(self):
        if self.lli is None:
            self.lli = {}
        if self.signal_strength is None:
            self.signal_strength = {}


class RINEXParser:
    """RINEX文件解析器"""

    # 卫星系统标识
    GNSS_SYSTEMS = {
        'G': 'GPS',
        'R': 'GLONASS',
        'E': 'Galileo',
        'C': 'BDS',  # BeiDou
        'J': 'QZSS',
        'I': 'IRNSS',
        'S': 'SBAS'
    }

    def __init__(self):
        self.header = None
        self.observations: List[Observation] = []

    def parse_file(self, file_path: str) -> Dict[str, Any]:
        """
        解析RINEX文件

        Args:
            file_path: RINEX文件路径（支持.gz压缩）

        Returns:
            解析结果字典
        """
        if not os.path.exists(file_path):
            raise RINEXError(f"File not found: {file_path}")

        logger.info(f"Parsing RINEX file: {file_path}")

        # 处理压缩文件
        if file_path.endswith('.gz'):
            with gzip.open(file_path, 'rt') as f:
                lines = f.readlines()
        else:
            with open(file_path, 'r') as f:
                lines = f.readlines()

        # 解析头部
        header_end = self._parse_header(lines)

        # 解析观测数据
        if self.header.file_type == 'O':
            self._parse_observations(lines[header_end:])

        result = {
            'file_path': file_path,
            'header': self.header,
            'observation_count': len(self.observations),
            'observations': self.observations
        }

        logger.info(f"Parsed {len(self.observations)} observation epochs")
        return result

    def _parse_header(self, lines: List[str]) -> int:
        """解析RINEX文件头部"""
        header_data = {
            'version': 2.11,
            'file_type': 'O',
            'satellite_system': 'G',
            'obs_types': []
        }

        line_idx = 0
        for i, line in enumerate(lines):
            if "END OF HEADER" in line:
                line_idx = i + 1
                break

            # RINEX版本和文件类型
            if "RINEX VERSION" in line:
                header_data['version'] = float(line[0:9].strip())
                header_data['file_type'] = line[20].strip()
                if len(line) > 40:
                    header_data['satellite_system'] = line[40].strip() or 'G'

            # 测站名称
            elif "MARKER NAME" in line:
                header_data['marker_name'] = line[0:60].strip()

            # 接收机类型
            elif "REC # / TYPE / VERS" in line:
                header_data['receiver_type'] = line[20:40].strip()

            # 天线类型
            elif "ANT # / TYPE" in line:
                header_data['antenna_type'] = line[20:40].strip()

            # 近似坐标
            elif "APPROX POSITION XYZ" in line:
                try:
                    x = float(line[0:14].strip())
                    y = float(line[14:28].strip())
                    z = float(line[28:42].strip())
                    header_data['approx_position'] = (x, y, z)
                except ValueError:
                    pass

            # 天线偏移
            elif "ANTENNA: DELTA H/E/N" in line:
                try:
                    h = float(line[0:14].strip())
                    e = float(line[14:28].strip())
                    n = float(line[28:42].strip())
                    header_data['antenna_delta'] = (h, e, n)
                except ValueError:
                    pass

            # 观测类型（RINEX 2.x）
            elif "# / TYPES OF OBSERV" in line:
                n_types = int(line[0:6].strip())
                types = line[10:60].split()
                header_data['obs_types'].extend(types)

            # 观测类型（RINEX 3.x）
            elif "SYS / # / OBS TYPES" in line:
                sys = line[0].strip()
                n_types = int(line[3:6].strip())
                types = line[7:60].split()
                header_data['obs_types'].extend([f"{sys}{t}" for t in types])

            # 采样间隔
            elif "INTERVAL" in line:
                try:
                    header_data['interval'] = float(line[0:10].strip())
                except ValueError:
                    pass

            # 首次观测时间
            elif "TIME OF FIRST OBS" in line:
                try:
                    year = int(line[0:6].strip())
                    month = int(line[6:12].strip())
                    day = int(line[12:18].strip())
                    hour = int(line[18:24].strip())
                    minute = int(line[24:30].strip())
                    second = float(line[30:43].strip())
                    header_data['time_of_first_obs'] = datetime(
                        year, month, day, hour, minute, int(second)
                    )
                except (ValueError, IndexError):
                    pass

            # 跳秒
            elif "LEAP SECONDS" in line:
                try:
                    header_data['leap_seconds'] = int(line[0:6].strip())
                except ValueError:
                    pass

        self.header = RINEXHeader(**header_data)
        return line_idx

    def _parse_observations(self, lines: List[str]):
        """解析观测数据记录"""
        i = 0
        while i < len(lines):
            line = lines[i]

            # RINEX 2.x 历元记录
            if self.header.version < 3.0:
                if len(line) > 26 and line[0] == ' ':
                    # 解析历元头
                    epoch_result = self._parse_epoch_v2(lines, i)
                    if epoch_result:
                        observations, next_line = epoch_result
                        self.observations.extend(observations)
                        i = next_line
                        continue

            # RINEX 3.x 历元记录
            else:
                if line.startswith('>'):
                    epoch_result = self._parse_epoch_v3(lines, i)
                    if epoch_result:
                        observations, next_line = epoch_result
                        self.observations.extend(observations)
                        i = next_line
                        continue

            i += 1

    def _parse_epoch_v2(self, lines: List[str], start_idx: int) -> Optional[Tuple[List[Observation], int]]:
        """解析RINEX 2.x历元数据"""
        line = lines[start_idx]

        try:
            # 解析历元时间
            year = int(line[1:3].strip())
            year = 2000 + year if year < 80 else 1900 + year
            month = int(line[4:6].strip())
            day = int(line[7:9].strip())
            hour = int(line[10:12].strip())
            minute = int(line[13:15].strip())
            second = float(line[16:26].strip())

            epoch = datetime(year, month, day, hour, minute, int(second))

            # 卫星数量
            n_sats = int(line[29:32].strip())

            # 读取卫星列表
            satellites = []
            sat_line_idx = start_idx
            chars_read = 32

            for _ in range(n_sats):
                if chars_read >= 68:  # 换行
                    sat_line_idx += 1
                    chars_read = 32
                    line = lines[sat_line_idx]

                sat = line[chars_read:chars_read+3].strip()
                satellites.append(sat)
                chars_read += 3

            # 解析每颗卫星的观测值
            observations = []
            data_line_idx = sat_line_idx + 1

            for sat in satellites:
                obs_data = {}
                line_idx = data_line_idx

                # 读取观测值（每行最多5个观测值）
                n_obs_types = len(self.header.obs_types)
                obs_read = 0

                while obs_read < n_obs_types:
                    if line_idx >= len(lines):
                        break

                    line = lines[line_idx]
                    obs_in_line = min(5, n_obs_types - obs_read)

                    for j in range(obs_in_line):
                        obs_type = self.header.obs_types[obs_read + j]
                        start_pos = j * 16
                        end_pos = start_pos + 14

                        obs_str = line[start_pos:end_pos].strip()
                        if obs_str:
                            try:
                                obs_data[obs_type] = float(obs_str)
                            except ValueError:
                                pass

                    obs_read += obs_in_line
                    line_idx += 1

                if obs_data:
                    observations.append(Observation(
                        epoch=epoch,
                        satellite=sat,
                        observations=obs_data
                    ))

                data_line_idx = line_idx

            return observations, data_line_idx

        except (ValueError, IndexError) as e:
            logger.warning(f"Failed to parse epoch at line {start_idx}: {e}")
            return None

    def _parse_epoch_v3(self, lines: List[str], start_idx: int) -> Optional[Tuple[List[Observation], int]]:
        """解析RINEX 3.x历元数据"""
        line = lines[start_idx]

        try:
            # 解析历元时间
            year = int(line[2:6].strip())
            month = int(line[7:9].strip())
            day = int(line[10:12].strip())
            hour = int(line[13:15].strip())
            minute = int(line[16:18].strip())
            second = float(line[19:29].strip())

            epoch = datetime(year, month, day, hour, minute, int(second))

            # 卫星数量
            n_sats = int(line[33:35].strip())

            # 解析每颗卫星的观测值
            observations = []
            line_idx = start_idx + 1

            for _ in range(n_sats):
                if line_idx >= len(lines):
                    break

                sat_line = lines[line_idx]
                satellite = sat_line[0:3].strip()

                obs_data = {}
                obs_types = [t for t in self.header.obs_types if t[0] == satellite[0]]

                # 读取观测值
                pos = 3
                for obs_type in obs_types:
                    if pos + 14 > len(sat_line):
                        break

                    obs_str = sat_line[pos:pos+14].strip()
                    if obs_str:
                        try:
                            obs_data[obs_type] = float(obs_str)
                        except ValueError:
                            pass

                    pos += 16

                if obs_data:
                    observations.append(Observation(
                        epoch=epoch,
                        satellite=satellite,
                        observations=obs_data
                    ))

                line_idx += 1

            return observations, line_idx

        except (ValueError, IndexError) as e:
            logger.warning(f"Failed to parse epoch at line {start_idx}: {e}")
            return None

    def extract_snr_data(self, satellite: Optional[str] = None) -> Dict[str, List[Tuple[datetime, float]]]:
        """
        提取SNR数据

        Args:
            satellite: 可选，指定卫星ID（如"G01"）

        Returns:
            字典，键为卫星ID，值为(时间, SNR)元组列表
        """
        snr_data = {}

        for obs in self.observations:
            if satellite and obs.satellite != satellite:
                continue

            # 查找SNR观测类型（S1, S2, S5等）
            snr_values = {k: v for k, v in obs.observations.items() if k.startswith('S')}

            if snr_values:
                if obs.satellite not in snr_data:
                    snr_data[obs.satellite] = []

                # 使用第一个可用的SNR值
                snr = list(snr_values.values())[0]
                snr_data[obs.satellite].append((obs.epoch, snr))

        return snr_data

    def get_observation_summary(self) -> Dict[str, Any]:
        """获取观测数据摘要"""
        if not self.observations:
            return {}

        satellites = set(obs.satellite for obs in self.observations)
        epochs = sorted(set(obs.epoch for obs in self.observations))

        obs_types_used = set()
        for obs in self.observations:
            obs_types_used.update(obs.observations.keys())

        return {
            'total_epochs': len(epochs),
            'total_satellites': len(satellites),
            'satellites': sorted(list(satellites)),
            'start_time': epochs[0] if epochs else None,
            'end_time': epochs[-1] if epochs else None,
            'duration_seconds': (epochs[-1] - epochs[0]).total_seconds() if len(epochs) > 1 else 0,
            'observation_types': sorted(list(obs_types_used)),
            'total_observations': len(self.observations)
        }


class RINEXConverter:
    """RINEX数据转换器"""

    @staticmethod
    def observations_to_dataframe(observations: List[Observation]):
        """将观测数据转换为pandas DataFrame"""
        try:
            import pandas as pd
        except ImportError:
            raise RINEXError("pandas is required for DataFrame conversion")

        data = []
        for obs in observations:
            for obs_type, value in obs.observations.items():
                data.append({
                    'epoch': obs.epoch,
                    'satellite': obs.satellite,
                    'obs_type': obs_type,
                    'value': value
                })

        return pd.DataFrame(data)

    @staticmethod
    def snr_to_arrays(snr_data: Dict[str, List[Tuple[datetime, float]]]) -> Dict[str, Dict[str, np.ndarray]]:
        """将SNR数据转换为numpy数组"""
        result = {}

        for sat, data in snr_data.items():
            if data:
                times, values = zip(*data)
                # 转换时间为秒数（从第一个时间点开始）
                time_seconds = np.array([(t - times[0]).total_seconds() for t in times])
                snr_values = np.array(values)

                result[sat] = {
                    'time': time_seconds,
                    'snr': snr_values,
                    'epochs': np.array(times)
                }

        return result


# 便捷函数

def parse_rinex_file(file_path: str) -> Dict[str, Any]:
    """
    解析RINEX文件

    Args:
        file_path: RINEX文件路径

    Returns:
        解析结果字典
    """
    parser = RINEXParser()
    return parser.parse_file(file_path)


def extract_snr_from_rinex(file_path: str, satellite: Optional[str] = None) -> Dict[str, List[Tuple[datetime, float]]]:
    """
    从RINEX文件提取SNR数据

    Args:
        file_path: RINEX文件路径
        satellite: 可选，指定卫星ID

    Returns:
        SNR数据字典
    """
    parser = RINEXParser()
    parser.parse_file(file_path)
    return parser.extract_snr_data(satellite=satellite)


if __name__ == "__main__":
    # 测试代码
    print("RINEX Parser module")
    print("\nSupported GNSS systems:")
    for code, name in RINEXParser.GNSS_SYSTEMS.items():
        print(f"  {code}: {name}")

    print("\nExample usage:")
    print("  parser = RINEXParser()")
    print("  result = parser.parse_file('station001.20o')")
    print("  snr_data = parser.extract_snr_data(satellite='G01')")
