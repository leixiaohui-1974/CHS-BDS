"""
系统监控模块

本模块提供系统运行监控功能：
- Prometheus风格的指标收集
- 系统资源监控（CPU、内存、磁盘）
- 应用性能指标
- 健康检查
- 指标导出

Author: Lei Xiaohui
Date: 2025-01-22
"""

import os
import time
import psutil
import platform
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict
from threading import Lock

from .logger import get_logger
from .exceptions import CHSBDSException

logger = get_logger(__name__)


class MonitoringError(CHSBDSException):
    """监控异常"""
    pass


@dataclass
class Metric:
    """监控指标"""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    unit: str = ""
    help_text: str = ""


class MetricsCollector:
    """指标收集器"""

    def __init__(self):
        self._metrics: Dict[str, List[Metric]] = defaultdict(list)
        self._lock = Lock()
        self._start_time = time.time()

    def record_metric(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
        unit: str = "",
        help_text: str = ""
    ):
        """记录指标"""
        metric = Metric(
            name=name,
            value=value,
            timestamp=datetime.now(),
            labels=labels or {},
            unit=unit,
            help_text=help_text
        )

        with self._lock:
            self._metrics[name].append(metric)

            # 保留最近1000个数据点
            if len(self._metrics[name]) > 1000:
                self._metrics[name] = self._metrics[name][-1000:]

    def get_metrics(self, name: Optional[str] = None) -> Dict[str, List[Metric]]:
        """获取指标"""
        with self._lock:
            if name:
                return {name: self._metrics.get(name, [])}
            return dict(self._metrics)

    def get_latest_metric(self, name: str) -> Optional[Metric]:
        """获取最新指标值"""
        with self._lock:
            metrics = self._metrics.get(name, [])
            return metrics[-1] if metrics else None

    def clear_metrics(self, name: Optional[str] = None):
        """清除指标"""
        with self._lock:
            if name:
                self._metrics.pop(name, None)
            else:
                self._metrics.clear()

    def get_uptime(self) -> float:
        """获取运行时间（秒）"""
        return time.time() - self._start_time


class SystemMonitor:
    """系统资源监控"""

    def __init__(self, collector: MetricsCollector):
        self.collector = collector
        self.process = psutil.Process()

    def collect_cpu_metrics(self):
        """收集CPU指标"""
        # 系统CPU使用率
        cpu_percent = psutil.cpu_percent(interval=0.1)
        self.collector.record_metric(
            "system_cpu_usage_percent",
            cpu_percent,
            unit="percent",
            help_text="System CPU usage percentage"
        )

        # 进程CPU使用率
        process_cpu = self.process.cpu_percent()
        self.collector.record_metric(
            "process_cpu_usage_percent",
            process_cpu,
            labels={"process": "chs_bds"},
            unit="percent",
            help_text="Process CPU usage percentage"
        )

        # CPU核心数
        cpu_count = psutil.cpu_count()
        self.collector.record_metric(
            "system_cpu_count",
            cpu_count,
            unit="cores",
            help_text="Number of CPU cores"
        )

    def collect_memory_metrics(self):
        """收集内存指标"""
        # 系统内存
        mem = psutil.virtual_memory()
        self.collector.record_metric(
            "system_memory_total_bytes",
            mem.total,
            unit="bytes",
            help_text="Total system memory"
        )

        self.collector.record_metric(
            "system_memory_used_bytes",
            mem.used,
            unit="bytes",
            help_text="Used system memory"
        )

        self.collector.record_metric(
            "system_memory_usage_percent",
            mem.percent,
            unit="percent",
            help_text="System memory usage percentage"
        )

        # 进程内存
        process_mem = self.process.memory_info()
        self.collector.record_metric(
            "process_memory_rss_bytes",
            process_mem.rss,
            labels={"process": "chs_bds"},
            unit="bytes",
            help_text="Process resident memory"
        )

        self.collector.record_metric(
            "process_memory_vms_bytes",
            process_mem.vms,
            labels={"process": "chs_bds"},
            unit="bytes",
            help_text="Process virtual memory"
        )

    def collect_disk_metrics(self):
        """收集磁盘指标"""
        # 磁盘使用情况
        disk = psutil.disk_usage('/')
        self.collector.record_metric(
            "system_disk_total_bytes",
            disk.total,
            labels={"mountpoint": "/"},
            unit="bytes",
            help_text="Total disk space"
        )

        self.collector.record_metric(
            "system_disk_used_bytes",
            disk.used,
            labels={"mountpoint": "/"},
            unit="bytes",
            help_text="Used disk space"
        )

        self.collector.record_metric(
            "system_disk_usage_percent",
            disk.percent,
            labels={"mountpoint": "/"},
            unit="percent",
            help_text="Disk usage percentage"
        )

        # IO统计
        try:
            io_counters = psutil.disk_io_counters()
            if io_counters:
                self.collector.record_metric(
                    "system_disk_read_bytes_total",
                    io_counters.read_bytes,
                    unit="bytes",
                    help_text="Total bytes read from disk"
                )

                self.collector.record_metric(
                    "system_disk_write_bytes_total",
                    io_counters.write_bytes,
                    unit="bytes",
                    help_text="Total bytes written to disk"
                )
        except:
            pass

    def collect_network_metrics(self):
        """收集网络指标"""
        try:
            net_io = psutil.net_io_counters()
            self.collector.record_metric(
                "system_network_bytes_sent_total",
                net_io.bytes_sent,
                unit="bytes",
                help_text="Total bytes sent over network"
            )

            self.collector.record_metric(
                "system_network_bytes_recv_total",
                net_io.bytes_recv,
                unit="bytes",
                help_text="Total bytes received over network"
            )

            self.collector.record_metric(
                "system_network_packets_sent_total",
                net_io.packets_sent,
                unit="packets",
                help_text="Total packets sent"
            )

            self.collector.record_metric(
                "system_network_packets_recv_total",
                net_io.packets_recv,
                unit="packets",
                help_text="Total packets received"
            )
        except:
            pass

    def collect_all_metrics(self):
        """收集所有系统指标"""
        self.collect_cpu_metrics()
        self.collect_memory_metrics()
        self.collect_disk_metrics()
        self.collect_network_metrics()


class ApplicationMonitor:
    """应用监控"""

    def __init__(self, collector: MetricsCollector):
        self.collector = collector
        self._request_count = defaultdict(int)
        self._request_duration = defaultdict(list)
        self._error_count = defaultdict(int)

    def record_request(self, endpoint: str, method: str = "GET"):
        """记录请求"""
        self._request_count[(endpoint, method)] += 1

        self.collector.record_metric(
            "http_requests_total",
            self._request_count[(endpoint, method)],
            labels={"endpoint": endpoint, "method": method},
            help_text="Total HTTP requests"
        )

    def record_request_duration(self, endpoint: str, duration: float, method: str = "GET"):
        """记录请求耗时"""
        self._request_duration[(endpoint, method)].append(duration)

        # 保留最近100个
        if len(self._request_duration[(endpoint, method)]) > 100:
            self._request_duration[(endpoint, method)] = \
                self._request_duration[(endpoint, method)][-100:]

        # 计算平均耗时
        avg_duration = sum(self._request_duration[(endpoint, method)]) / \
                      len(self._request_duration[(endpoint, method)])

        self.collector.record_metric(
            "http_request_duration_seconds",
            avg_duration,
            labels={"endpoint": endpoint, "method": method},
            unit="seconds",
            help_text="Average HTTP request duration"
        )

    def record_error(self, error_type: str, module: str = "unknown"):
        """记录错误"""
        self._error_count[(error_type, module)] += 1

        self.collector.record_metric(
            "errors_total",
            self._error_count[(error_type, module)],
            labels={"type": error_type, "module": module},
            help_text="Total errors"
        )

    def record_analysis_run(self, module: str, duration: float, success: bool):
        """记录分析运行"""
        status = "success" if success else "failure"

        self.collector.record_metric(
            "analysis_runs_total",
            1,
            labels={"module": module, "status": status},
            help_text="Total analysis runs"
        )

        self.collector.record_metric(
            "analysis_duration_seconds",
            duration,
            labels={"module": module},
            unit="seconds",
            help_text="Analysis duration"
        )


class HealthChecker:
    """健康检查"""

    def __init__(self):
        self.checks: Dict[str, callable] = {}

    def register_check(self, name: str, check_func: callable):
        """注册健康检查"""
        self.checks[name] = check_func

    def run_checks(self) -> Dict[str, Any]:
        """运行所有健康检查"""
        results = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'checks': {}
        }

        all_healthy = True

        for name, check_func in self.checks.items():
            try:
                check_result = check_func()
                results['checks'][name] = {
                    'status': 'pass' if check_result else 'fail',
                    'healthy': check_result
                }

                if not check_result:
                    all_healthy = False

            except Exception as e:
                results['checks'][name] = {
                    'status': 'error',
                    'healthy': False,
                    'error': str(e)
                }
                all_healthy = False

        results['status'] = 'healthy' if all_healthy else 'unhealthy'
        return results


class MonitoringService:
    """监控服务"""

    def __init__(self):
        self.collector = MetricsCollector()
        self.system_monitor = SystemMonitor(self.collector)
        self.app_monitor = ApplicationMonitor(self.collector)
        self.health_checker = HealthChecker()

        # 注册默认健康检查
        self._register_default_health_checks()

    def _register_default_health_checks(self):
        """注册默认健康检查"""

        def check_cpu():
            cpu_percent = psutil.cpu_percent(interval=0.1)
            return cpu_percent < 90.0

        def check_memory():
            mem = psutil.virtual_memory()
            return mem.percent < 90.0

        def check_disk():
            disk = psutil.disk_usage('/')
            return disk.percent < 90.0

        self.health_checker.register_check("cpu", check_cpu)
        self.health_checker.register_check("memory", check_memory)
        self.health_checker.register_check("disk", check_disk)

    def collect_metrics(self):
        """收集所有指标"""
        self.system_monitor.collect_all_metrics()

        # 记录运行时间
        uptime = self.collector.get_uptime()
        self.collector.record_metric(
            "system_uptime_seconds",
            uptime,
            unit="seconds",
            help_text="System uptime"
        )

    def get_prometheus_metrics(self) -> str:
        """获取Prometheus格式的指标"""
        lines = []
        metrics = self.collector.get_metrics()

        for name, metric_list in metrics.items():
            if not metric_list:
                continue

            latest = metric_list[-1]

            # HELP
            if latest.help_text:
                lines.append(f"# HELP {name} {latest.help_text}")

            # TYPE
            lines.append(f"# TYPE {name} gauge")

            # Metric
            if latest.labels:
                label_str = ','.join([f'{k}="{v}"' for k, v in latest.labels.items()])
                lines.append(f"{name}{{{label_str}}} {latest.value}")
            else:
                lines.append(f"{name} {latest.value}")

        return '\n'.join(lines)

    def get_metrics_summary(self) -> Dict[str, Any]:
        """获取指标摘要"""
        summary = {}

        # CPU
        cpu_metric = self.collector.get_latest_metric("system_cpu_usage_percent")
        if cpu_metric:
            summary['cpu_usage_percent'] = cpu_metric.value

        # 内存
        mem_metric = self.collector.get_latest_metric("system_memory_usage_percent")
        if mem_metric:
            summary['memory_usage_percent'] = mem_metric.value

        # 磁盘
        disk_metric = self.collector.get_latest_metric("system_disk_usage_percent")
        if disk_metric:
            summary['disk_usage_percent'] = disk_metric.value

        # 运行时间
        uptime_metric = self.collector.get_latest_metric("system_uptime_seconds")
        if uptime_metric:
            summary['uptime_seconds'] = uptime_metric.value

        return summary


# 全局监控服务实例
_monitoring_service = None


def get_monitoring_service() -> MonitoringService:
    """获取监控服务单例"""
    global _monitoring_service
    if _monitoring_service is None:
        _monitoring_service = MonitoringService()
    return _monitoring_service


if __name__ == "__main__":
    print("System Monitoring module")
    print(f"\nSystem Information:")
    print(f"  Platform: {platform.system()} {platform.release()}")
    print(f"  CPU Cores: {psutil.cpu_count()}")
    print(f"  Total Memory: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    print(f"  Total Disk: {psutil.disk_usage('/').total / (1024**3):.2f} GB")

    # 测试监控服务
    service = get_monitoring_service()
    service.collect_metrics()

    print(f"\nCurrent Metrics:")
    summary = service.get_metrics_summary()
    for key, value in summary.items():
        print(f"  {key}: {value:.2f}")
