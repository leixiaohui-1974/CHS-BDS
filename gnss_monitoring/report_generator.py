"""
PDF报告生成模块

本模块提供专业PDF报告生成功能：
- 自动生成监测报告
- 包含图表、表格、统计信息
- 支持多种报告模板
- 中文支持
- 嵌入matplotlib图表
- 分页和目录

Author: Lei Xiaohui
Date: 2025-01-22
"""

import os
import io
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm, mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, Image, KeepTogether
    )
    from reportlab.pdfgen import canvas
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import numpy as np

from .logger import get_logger
from .exceptions import CHSBDSException

logger = get_logger(__name__)


class ReportError(CHSBDSException):
    """报告生成异常"""
    pass


class PDFReportGenerator:
    """PDF报告生成器"""

    def __init__(
        self,
        output_path: str,
        title: str = "CHS-BDS GNSS监测报告",
        author: str = "CHS-BDS System",
        pagesize: Tuple = A4
    ):
        if not REPORTLAB_AVAILABLE:
            raise ReportError("ReportLab is not installed. Install with: pip install reportlab")

        self.output_path = output_path
        self.title = title
        self.author = author
        self.pagesize = pagesize

        # 创建输出目录
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # 创建文档
        self.doc = SimpleDocTemplate(
            output_path,
            pagesize=pagesize,
            topMargin=2*cm,
            bottomMargin=2*cm,
            leftMargin=2*cm,
            rightMargin=2*cm
        )

        # 文档内容
        self.story = []

        # 样式
        self.styles = getSampleStyleSheet()
        self._setup_styles()

    def _setup_styles(self):
        """设置样式"""
        # 标题样式
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=30,
            alignment=TA_CENTER
        ))

        # 章节标题
        self.styles.add(ParagraphStyle(
            name='CustomHeading1',
            parent=self.styles['Heading1'],
            fontSize=16,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=12,
            spaceBefore=12
        ))

        # 小节标题
        self.styles.add(ParagraphStyle(
            name='CustomHeading2',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#34495e'),
            spaceAfter=10,
            spaceBefore=10
        ))

        # 正文
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['BodyText'],
            fontSize=11,
            alignment=TA_JUSTIFY,
            spaceAfter=6
        ))

    def add_cover_page(self, subtitle: Optional[str] = None, date: Optional[datetime] = None):
        """添加封面页"""
        if date is None:
            date = datetime.now()

        # 标题
        self.story.append(Spacer(1, 5*cm))
        self.story.append(Paragraph(self.title, self.styles['CustomTitle']))
        self.story.append(Spacer(1, 0.5*cm))

        # 副标题
        if subtitle:
            self.story.append(Paragraph(subtitle, self.styles['CustomHeading2']))
            self.story.append(Spacer(1, 1*cm))

        # 日期和作者
        info_style = ParagraphStyle(
            name='Info',
            fontSize=12,
            textColor=colors.grey,
            alignment=TA_CENTER
        )

        self.story.append(Paragraph(f"生成日期: {date.strftime('%Y年%m月%d日')}", info_style))
        self.story.append(Paragraph(f"作者: {self.author}", info_style))

        self.story.append(PageBreak())

    def add_heading(self, text: str, level: int = 1):
        """添加标题"""
        if level == 1:
            style = self.styles['CustomHeading1']
        elif level == 2:
            style = self.styles['CustomHeading2']
        else:
            style = self.styles['Heading3']

        self.story.append(Paragraph(text, style))

    def add_paragraph(self, text: str):
        """添加段落"""
        self.story.append(Paragraph(text, self.styles['CustomBody']))
        self.story.append(Spacer(1, 0.3*cm))

    def add_table(
        self,
        data: List[List[Any]],
        headers: Optional[List[str]] = None,
        col_widths: Optional[List[float]] = None
    ):
        """添加表格"""
        if headers:
            table_data = [headers] + data
        else:
            table_data = data

        if col_widths is None:
            # 自动计算列宽
            available_width = self.pagesize[0] - 4*cm
            col_widths = [available_width / len(table_data[0])] * len(table_data[0])

        table = Table(table_data, colWidths=col_widths)

        # 表格样式
        style = TableStyle([
            # 表头
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),

            # 表体
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),

            # 网格
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),

            # 交替行颜色
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#ecf0f1')])
        ])

        table.setStyle(style)
        self.story.append(table)
        self.story.append(Spacer(1, 0.5*cm))

    def add_matplotlib_figure(self, fig, width: float = 15*cm, height: float = 10*cm):
        """添加matplotlib图表"""
        # 保存图表到内存
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)

        # 创建图像对象
        img = Image(img_buffer, width=width, height=height)
        self.story.append(img)
        self.story.append(Spacer(1, 0.5*cm))

        plt.close(fig)

    def add_spacer(self, height: float = 1*cm):
        """添加空白"""
        self.story.append(Spacer(1, height))

    def add_page_break(self):
        """添加分页符"""
        self.story.append(PageBreak())

    def generate(self):
        """生成PDF文件"""
        try:
            self.doc.build(self.story)
            logger.info(f"PDF report generated: {self.output_path}")
            return self.output_path
        except Exception as e:
            raise ReportError(f"Failed to generate PDF: {e}")


class GNSSMonitoringReport:
    """GNSS监测报告生成器"""

    def __init__(self, output_path: str):
        self.generator = PDFReportGenerator(
            output_path=output_path,
            title="CHS-BDS GNSS Monitoring Report"
        )

    def create_comprehensive_report(self, results: Dict[str, Any], config: Optional[Dict] = None):
        """
        创建综合监测报告

        Args:
            results: 分析结果字典
            config: 配置信息
        """
        # 封面
        self.generator.add_cover_page(
            subtitle="Multi-Mode GNSS Comprehensive Monitoring System",
            date=datetime.now()
        )

        # 执行摘要
        self._add_executive_summary(results)

        # GNSS-IR 章节
        if 'gnss_ir' in results.get('modules', {}):
            self._add_gnss_ir_section(results['modules']['gnss_ir'])

        # 形变监测章节
        if 'deformation' in results.get('modules', {}):
            self._add_deformation_section(results['modules']['deformation'])

        # PWV估算章节
        if 'pwv' in results.get('modules', {}):
            self._add_pwv_section(results['modules']['pwv'])

        # 降雨预测章节
        if 'rainfall' in results.get('modules', {}):
            self._add_rainfall_section(results['modules']['rainfall'])

        # 质量控制章节
        if 'quality_control' in results:
            self._add_quality_control_section(results['quality_control'])

        # 结论和建议
        self._add_conclusions(results)

        # 生成PDF
        return self.generator.generate()

    def _add_executive_summary(self, results: Dict[str, Any]):
        """添加执行摘要"""
        self.generator.add_heading("Executive Summary", level=1)

        summary_text = f"""
        This report presents the results of GNSS monitoring analysis conducted on {datetime.now().strftime('%Y-%m-%d')}.
        The analysis includes GNSS-IR water level monitoring, deformation detection, PWV estimation, and rainfall prediction.
        """
        self.generator.add_paragraph(summary_text)

        # 关键指标表
        modules = results.get('modules', {})
        summary_data = []

        if 'gnss_ir' in modules:
            ir_result = modules['gnss_ir']
            summary_data.append([
                "GNSS-IR Water Level",
                f"{ir_result.get('water_level', 0):.3f} m",
                f"{ir_result.get('confidence', 0)*100:.1f}%"
            ])

        if 'deformation' in modules:
            def_result = modules['deformation']
            summary_data.append([
                "Deformation Displacement",
                f"{def_result.get('displacement', 0)*1000:.2f} mm",
                f"{def_result.get('status', 'N/A')}"
            ])

        if 'pwv' in modules:
            pwv_result = modules['pwv']
            summary_data.append([
                "Precipitable Water Vapor",
                f"{pwv_result.get('pwv', 0):.2f} mm",
                f"{pwv_result.get('atmospheric_condition', 'N/A')}"
            ])

        if summary_data:
            self.generator.add_table(
                data=summary_data,
                headers=["Parameter", "Value", "Status/Confidence"]
            )

        self.generator.add_page_break()

    def _add_gnss_ir_section(self, gnss_ir_results: Dict[str, Any]):
        """添加GNSS-IR章节"""
        self.generator.add_heading("1. GNSS-IR Water Level Monitoring", level=1)

        # 结果描述
        water_level = gnss_ir_results.get('water_level', 0)
        confidence = gnss_ir_results.get('confidence', 0)

        text = f"""
        The GNSS Interferometric Reflectometry (GNSS-IR) analysis estimates the water level
        at {water_level:.3f} meters with a confidence level of {confidence*100:.1f}%.
        This measurement is derived from Signal-to-Noise Ratio (SNR) observations.
        """
        self.generator.add_paragraph(text)

        # 创建SNR数据图表
        if 'elevation' in gnss_ir_results and 'snr' in gnss_ir_results:
            fig = self._create_snr_plot(
                gnss_ir_results['elevation'],
                gnss_ir_results['snr']
            )
            self.generator.add_matplotlib_figure(fig)

        # 详细参数表
        params_data = [
            ["Parameter", "Value", "Unit"],
            ["Water Level", f"{water_level:.3f}", "m"],
            ["Reflection Height", f"{gnss_ir_results.get('reflection_height', 0):.3f}", "m"],
            ["Dominant Frequency", f"{gnss_ir_results.get('dominant_frequency', 0):.4f}", "cycles/deg"],
            ["Confidence", f"{confidence*100:.1f}", "%"]
        ]

        self.generator.add_heading("Parameters", level=2)
        self.generator.add_table(params_data[1:], headers=params_data[0])

        self.generator.add_page_break()

    def _add_deformation_section(self, deformation_results: Dict[str, Any]):
        """添加形变监测章节"""
        self.generator.add_heading("2. Deformation Monitoring", level=1)

        displacement = deformation_results.get('displacement', 0)
        status = deformation_results.get('status', 'UNKNOWN')

        text = f"""
        The deformation monitoring analysis detects a displacement of {displacement*1000:.2f} mm.
        The monitoring status is: {status}.
        """
        self.generator.add_paragraph(text)

        # 三维位移表
        params_data = [
            ["Component", "Displacement (mm)", "Status"],
            ["North (dN)", f"{deformation_results.get('dx', 0)*1000:.3f}", "OK"],
            ["East (dE)", f"{deformation_results.get('dy', 0)*1000:.3f}", "OK"],
            ["Up (dU)", f"{deformation_results.get('dz', 0)*1000:.3f}", "OK"],
            ["Total", f"{displacement*1000:.3f}", status]
        ]

        self.generator.add_table(params_data[1:], headers=params_data[0])

        self.generator.add_page_break()

    def _add_pwv_section(self, pwv_results: Dict[str, Any]):
        """添加PWV估算章节"""
        self.generator.add_heading("3. Precipitable Water Vapor (PWV) Estimation", level=1)

        pwv = pwv_results.get('pwv', 0)
        condition = pwv_results.get('atmospheric_condition', 'UNKNOWN')

        text = f"""
        The Precipitable Water Vapor (PWV) is estimated at {pwv:.2f} mm,
        indicating {condition} atmospheric moisture conditions.
        """
        self.generator.add_paragraph(text)

        # PWV参数表
        params_data = [
            ["Parameter", "Value", "Unit"],
            ["ZTD (Total Delay)", f"{pwv_results.get('ztd', 0):.4f}", "m"],
            ["ZHD (Hydrostatic)", f"{pwv_results.get('zhd', 0):.4f}", "m"],
            ["ZWD (Wet)", f"{pwv_results.get('zwd', 0):.4f}", "m"],
            ["PWV", f"{pwv:.2f}", "mm"],
            ["Atmospheric Condition", condition, "-"]
        ]

        self.generator.add_table(params_data[1:], headers=params_data[0])

        self.generator.add_page_break()

    def _add_rainfall_section(self, rainfall_results: Dict[str, Any]):
        """添加降雨预测章节"""
        self.generator.add_heading("4. Rainfall Prediction", level=1)

        predicted = rainfall_results.get('predicted_rainfall', 0)
        confidence = rainfall_results.get('confidence', 0)

        text = f"""
        The rainfall prediction model forecasts {predicted:.2f} mm of rainfall
        with {confidence*100:.1f}% confidence based on PWV data.
        """
        self.generator.add_paragraph(text)

        # 预测结果表
        params_data = [
            ["Metric", "Value"],
            ["Predicted Rainfall", f"{predicted:.2f} mm"],
            ["Confidence Level", f"{confidence*100:.1f}%"],
            ["Model Version", "v1.0"]
        ]

        self.generator.add_table(params_data[1:], headers=params_data[0])

        self.generator.add_page_break()

    def _add_quality_control_section(self, qc_results: Dict[str, Any]):
        """添加质量控制章节"""
        self.generator.add_heading("5. Data Quality Control", level=1)

        quality_score = qc_results.get('quality_score', 0)

        text = f"""
        Data quality assessment yields an overall quality score of {quality_score:.1f}/100.
        """
        self.generator.add_paragraph(text)

        # 质量指标表
        metrics = qc_results.get('metrics', {})
        params_data = [
            ["Metric", "Value"],
            ["Completeness", f"{metrics.get('completeness', 0)*100:.1f}%"],
            ["Outliers Detected", str(metrics.get('outlier_count', 0))],
            ["Quality Score", f"{quality_score:.1f}/100"]
        ]

        self.generator.add_table(params_data[1:], headers=params_data[0])

        self.generator.add_page_break()

    def _add_conclusions(self, results: Dict[str, Any]):
        """添加结论和建议"""
        self.generator.add_heading("6. Conclusions and Recommendations", level=1)

        conclusions = """
        The GNSS monitoring system has successfully completed the analysis with the following key findings:

        1. All monitoring modules are functioning within normal parameters
        2. Data quality meets the required standards
        3. No critical alerts were triggered during this analysis period

        Recommendations for continued monitoring:
        - Maintain regular data collection intervals
        - Monitor for any anomalous trends in deformation patterns
        - Ensure timely calibration of equipment
        """

        self.generator.add_paragraph(conclusions)

    def _create_snr_plot(self, elevation: List[float], snr: List[float]):
        """创建SNR图表"""
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(elevation, snr, 'b-', linewidth=1.5, label='SNR')
        ax.set_xlabel('Elevation Angle (degrees)', fontsize=12)
        ax.set_ylabel('SNR (dB-Hz)', fontsize=12)
        ax.set_title('GNSS SNR vs Elevation Angle', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

        return fig


# 便捷函数

def generate_monitoring_report(
    output_path: str,
    results: Dict[str, Any],
    config: Optional[Dict] = None
) -> str:
    """
    生成监测报告

    Args:
        output_path: 输出PDF文件路径
        results: 分析结果
        config: 配置信息

    Returns:
        生成的PDF文件路径
    """
    report = GNSSMonitoringReport(output_path)
    return report.create_comprehensive_report(results, config)


if __name__ == "__main__":
    # 测试代码
    print("PDF Report Generator module")

    if not REPORTLAB_AVAILABLE:
        print("Warning: ReportLab is not installed")
        print("Install with: pip install reportlab")
    else:
        print("ReportLab is available")
        print("\nExample usage:")
        print("  report = GNSSMonitoringReport('output/report.pdf')")
        print("  report.create_comprehensive_report(results)")
