"""
高级可视化模块

本模块提供高级数据可视化功能：
- 3D轨迹图
- 交互式时间序列
- 地图可视化
- 热力图
- 动画
- 多图表组合

Author: Lei Xiaohui
Date: 2025-01-22
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from .logger import get_logger
from .exceptions import CHSBDSException

logger = get_logger(__name__)


class VisualizationError(CHSBDSException):
    """可视化异常"""
    pass


class Advanced3DVisualizer:
    """3D可视化器"""

    @staticmethod
    def plot_3d_trajectory(
        coordinates: List[Tuple[float, float, float]],
        title: str = "3D Trajectory",
        labels: Optional[List[str]] = None
    ):
        """
        绘制3D轨迹图

        Args:
            coordinates: 坐标列表 [(x, y, z), ...]
            title: 图表标题
            labels: 可选的点标签
        """
        if not coordinates:
            raise VisualizationError("No coordinates provided")

        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        # 提取坐标
        x, y, z = zip(*coordinates)

        # 绘制轨迹
        ax.plot(x, y, z, 'b-', linewidth=2, alpha=0.6, label='Trajectory')
        ax.scatter(x, y, z, c='red', marker='o', s=50, alpha=0.8)

        # 起点和终点
        ax.scatter([x[0]], [y[0]], [z[0]], c='green', marker='*', s=200, label='Start')
        ax.scatter([x[-1]], [y[-1]], [z[-1]], c='red', marker='*', s=200, label='End')

        # 标签
        if labels:
            for i, label in enumerate(labels):
                if i < len(coordinates):
                    ax.text(x[i], y[i], z[i], label, fontsize=8)

        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_zlabel('Z (m)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_deformation_3d(
        station_coords: Dict[str, Tuple[float, float, float]],
        displacements: Dict[str, Tuple[float, float, float]],
        scale: float = 1000.0
    ):
        """
        绘制形变3D向量图

        Args:
            station_coords: 测站坐标 {station_id: (x, y, z)}
            displacements: 位移向量 {station_id: (dx, dy, dz)}
            scale: 位移缩放比例
        """
        if not PLOTLY_AVAILABLE:
            raise VisualizationError("Plotly is required for this visualization")

        fig = go.Figure()

        # 绘制测站
        for station_id, (x, y, z) in station_coords.items():
            dx, dy, dz = displacements.get(station_id, (0, 0, 0))

            # 测站位置
            fig.add_trace(go.Scatter3d(
                x=[x],
                y=[y],
                z=[z],
                mode='markers+text',
                name=station_id,
                text=[station_id],
                marker=dict(size=8, color='blue')
            ))

            # 位移向量
            if dx != 0 or dy != 0 or dz != 0:
                fig.add_trace(go.Scatter3d(
                    x=[x, x + dx * scale],
                    y=[y, y + dy * scale],
                    z=[z, z + dz * scale],
                    mode='lines',
                    name=f'{station_id} displacement',
                    line=dict(color='red', width=4),
                    showlegend=False
                ))

        fig.update_layout(
            title="3D Deformation Visualization",
            scene=dict(
                xaxis_title="X (m)",
                yaxis_title="Y (m)",
                zaxis_title="Z (m)"
            ),
            width=900,
            height=700
        )

        return fig


class InteractiveTimeSeriesVisualizer:
    """交互式时间序列可视化器"""

    @staticmethod
    def plot_multi_series(
        data: Dict[str, List[Tuple[datetime, float]]],
        title: str = "Time Series",
        y_label: str = "Value"
    ):
        """
        绘制多个时间序列

        Args:
            data: 时间序列数据 {series_name: [(datetime, value), ...]}
            title: 图表标题
            y_label: Y轴标签
        """
        if not PLOTLY_AVAILABLE:
            raise VisualizationError("Plotly is required for interactive plots")

        fig = go.Figure()

        for series_name, series_data in data.items():
            if series_data:
                times, values = zip(*series_data)

                fig.add_trace(go.Scatter(
                    x=times,
                    y=values,
                    mode='lines+markers',
                    name=series_name,
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                  'Time: %{x}<br>' +
                                  'Value: %{y:.4f}<extra></extra>'
                ))

        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title=y_label,
            hovermode='x unified',
            width=1000,
            height=600,
            showlegend=True
        )

        return fig

    @staticmethod
    def plot_with_anomalies(
        timestamps: List[datetime],
        values: List[float],
        anomalies: List[int],
        title: str = "Time Series with Anomalies"
    ):
        """
        绘制带异常标记的时间序列

        Args:
            timestamps: 时间戳列表
            values: 数值列表
            anomalies: 异常点索引列表
            title: 标题
        """
        if not PLOTLY_AVAILABLE:
            raise VisualizationError("Plotly is required")

        fig = go.Figure()

        # 正常数据
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=values,
            mode='lines+markers',
            name='Normal',
            line=dict(color='blue'),
            marker=dict(size=4)
        ))

        # 异常点
        if anomalies:
            anomaly_times = [timestamps[i] for i in anomalies if i < len(timestamps)]
            anomaly_values = [values[i] for i in anomalies if i < len(values)]

            fig.add_trace(go.Scatter(
                x=anomaly_times,
                y=anomaly_values,
                mode='markers',
                name='Anomalies',
                marker=dict(color='red', size=10, symbol='x')
            ))

        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Value",
            width=1000,
            height=600
        )

        return fig


class HeatmapVisualizer:
    """热力图可视化器"""

    @staticmethod
    def plot_correlation_heatmap(
        data: Dict[str, List[float]],
        title: str = "Correlation Heatmap"
    ):
        """
        绘制相关性热力图

        Args:
            data: 数据字典 {variable_name: [values]}
            title: 标题
        """
        if not PLOTLY_AVAILABLE:
            raise VisualizationError("Plotly is required")

        import pandas as pd

        # 创建DataFrame并计算相关性
        df = pd.DataFrame(data)
        corr_matrix = df.corr()

        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))

        fig.update_layout(
            title=title,
            width=800,
            height=700
        )

        return fig

    @staticmethod
    def plot_spatial_heatmap(
        x: List[float],
        y: List[float],
        values: List[float],
        title: str = "Spatial Heatmap"
    ):
        """
        绘制空间热力图

        Args:
            x: X坐标
            y: Y坐标
            values: 数值
            title: 标题
        """
        if not PLOTLY_AVAILABLE:
            raise VisualizationError("Plotly is required")

        fig = go.Figure(data=go.Scatter(
            x=x,
            y=y,
            mode='markers',
            marker=dict(
                size=15,
                color=values,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Value")
            ),
            text=[f"Value: {v:.2f}" for v in values],
            hovertemplate='X: %{x}<br>Y: %{y}<br>%{text}<extra></extra>'
        ))

        fig.update_layout(
            title=title,
            xaxis_title="X",
            yaxis_title="Y",
            width=800,
            height=700
        )

        return fig


class MultiPanelVisualizer:
    """多面板可视化器"""

    @staticmethod
    def create_gnss_monitoring_dashboard(results: Dict[str, Any]):
        """
        创建GNSS监测仪表板

        Args:
            results: 监测结果字典
        """
        if not PLOTLY_AVAILABLE:
            raise VisualizationError("Plotly is required")

        # 创建子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'GNSS-IR Water Level',
                'Deformation Displacement',
                'PWV Estimation',
                'Rainfall Prediction'
            ),
            specs=[
                [{'type': 'scatter'}, {'type': 'scatter'}],
                [{'type': 'scatter'}, {'type': 'bar'}]
            ]
        )

        # GNSS-IR (示例数据)
        if 'gnss_ir' in results.get('modules', {}):
            ir_data = results['modules']['gnss_ir']
            fig.add_trace(
                go.Scatter(
                    x=[datetime.now()],
                    y=[ir_data.get('water_level', 0)],
                    mode='markers',
                    name='Water Level',
                    marker=dict(size=15, color='blue')
                ),
                row=1, col=1
            )

        # Deformation
        if 'deformation' in results.get('modules', {}):
            def_data = results['modules']['deformation']
            displacement = def_data.get('displacement', 0) * 1000  # 转换为mm

            fig.add_trace(
                go.Scatter(
                    x=[datetime.now()],
                    y=[displacement],
                    mode='markers',
                    name='Displacement',
                    marker=dict(size=15, color='red')
                ),
                row=1, col=2
            )

        # PWV
        if 'pwv' in results.get('modules', {}):
            pwv_data = results['modules']['pwv']

            fig.add_trace(
                go.Scatter(
                    x=[datetime.now()],
                    y=[pwv_data.get('pwv', 0)],
                    mode='markers',
                    name='PWV',
                    marker=dict(size=15, color='green')
                ),
                row=2, col=1
            )

        # Rainfall
        if 'rainfall' in results.get('modules', {}):
            rainfall_data = results['modules']['rainfall']

            fig.add_trace(
                go.Bar(
                    x=['Predicted'],
                    y=[rainfall_data.get('predicted_rainfall', 0)],
                    name='Rainfall',
                    marker_color='lightblue'
                ),
                row=2, col=2
            )

        # 更新布局
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_xaxes(title_text="Time", row=1, col=2)
        fig.update_xaxes(title_text="Time", row=2, col=1)

        fig.update_yaxes(title_text="Water Level (m)", row=1, col=1)
        fig.update_yaxes(title_text="Displacement (mm)", row=1, col=2)
        fig.update_yaxes(title_text="PWV (mm)", row=2, col=1)
        fig.update_yaxes(title_text="Rainfall (mm)", row=2, col=2)

        fig.update_layout(
            title_text="GNSS Monitoring Dashboard",
            height=800,
            showlegend=False
        )

        return fig


# 便捷函数

def save_plotly_figure(fig, output_path: str, format: str = 'html'):
    """
    保存Plotly图表

    Args:
        fig: Plotly图表对象
        output_path: 输出路径
        format: 格式 ('html', 'png', 'jpg', 'svg', 'pdf')
    """
    if format == 'html':
        fig.write_html(output_path)
    else:
        fig.write_image(output_path, format=format)

    logger.info(f"Figure saved to: {output_path}")


if __name__ == "__main__":
    print("Advanced Visualization module")

    if not PLOTLY_AVAILABLE:
        print("Warning: Plotly is not installed")
        print("Install with: pip install plotly kaleido")
    else:
        print("Plotly is available")
        print("\nExample usage:")
        print("  visualizer = Advanced3DVisualizer()")
        print("  fig = visualizer.plot_3d_trajectory(coordinates)")
        print("  save_plotly_figure(fig, 'output.html')")
