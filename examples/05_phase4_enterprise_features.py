"""
Phase 4企业级功能演示

本示例展示CHS-BDS Phase 4的所有企业级功能：
1. 数据库集成 (PostgreSQL/TimescaleDB)
2. RESTful API服务 (FastAPI)
3. RINEX文件解析
4. PDF报告生成
5. 系统监控
6. 用户认证

Author: Lei Xiaohui
Date: 2025-01-22
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import gnss_monitoring as gnss


def demo_database_integration():
    """演示数据库集成"""
    print("=" * 80)
    print("1. Database Integration Demo")
    print("=" * 80)

    # 创建数据库配置
    db_config = gnss.DatabaseConfig(
        host="localhost",
        port=5432,
        database="chs_bds_test",
        user="postgres",
        password="postgres",
        enable_timescaledb=True
    )

    print(f"\n数据库配置:")
    print(f"  Host: {db_config.host}")
    print(f"  Database: {db_config.database}")
    print(f"  TimescaleDB: {db_config.enable_timescaledb}")

    # 注意：实际运行需要PostgreSQL服务器
    print("\n⚠️  需要PostgreSQL服务器才能运行此功能")
    print("  安装: sudo apt-get install postgresql")
    print("  配置: 创建数据库和用户")

    # 示例：数据插入
    print("\n数据插入示例:")
    print("""
    # 初始化数据库
    db_manager = gnss.DatabaseManager(db_config)
    db_manager.initialize()

    # 插入GNSS-IR数据
    gnss_ir_data = [{
        'timestamp': datetime.now(),
        'station_id': 'STAT001',
        'satellite_id': 'G01',
        'water_level': 2.456,
        'quality_score': 0.95,
        'metadata': {'note': 'Good quality'}
    }]
    db_manager.insert_gnss_ir_data(gnss_ir_data)

    # 查询数据
    results = db_manager.query_gnss_ir_data(
        station_id='STAT001',
        start_time=datetime.now() - timedelta(days=7)
    )
    print(f"查询到 {len(results)} 条记录")
    """)


def demo_rinex_parser():
    """演示RINEX文件解析"""
    print("\n" + "=" * 80)
    print("2. RINEX File Parser Demo")
    print("=" * 80)

    print("\nRINEX解析器支持:")
    print("  - RINEX 2.x 和 3.x 格式")
    print("  - GPS, GLONASS, Galileo, BeiDou等多系统")
    print("  - 自动SNR数据提取")
    print("  - 支持.gz压缩文件")

    print("\n使用示例:")
    print("""
    # 解析RINEX观测文件
    parser = gnss.RINEXParser()
    result = parser.parse_file('station001.20o')

    print(f"RINEX版本: {result['header'].version}")
    print(f"测站名称: {result['header'].marker_name}")
    print(f"观测历元数: {result['observation_count']}")

    # 提取SNR数据
    snr_data = parser.extract_snr_data(satellite='G01')

    # 获取观测摘要
    summary = parser.get_observation_summary()
    print(f"观测时段: {summary['start_time']} ~ {summary['end_time']}")
    print(f"观测卫星: {', '.join(summary['satellites'])}")
    """)

    print("\n⚠️  需要实际的RINEX文件进行测试")


def demo_pdf_report_generation():
    """演示PDF报告生成"""
    print("\n" + "=" * 80)
    print("3. PDF Report Generation Demo")
    print("=" * 80)

    print("\nPDF报告功能:")
    print("  - 专业格式报告")
    print("  - 自动图表嵌入")
    print("  - 表格和统计信息")
    print("  - 分章节组织")

    print("\n生成监测报告示例:")

    # 准备测试数据
    test_results = {
        'timestamp': datetime.now(),
        'modules': {
            'gnss_ir': {
                'water_level': 2.456,
                'confidence': 0.95,
                'reflection_height': 2.5,
                'dominant_frequency': 0.0234,
                'elevation': list(range(5, 90, 5)),
                'snr': [45 + i * 0.5 for i in range(17)]
            },
            'deformation': {
                'displacement': 0.0234,  # 23.4 mm
                'dx': 0.012,
                'dy': 0.008,
                'dz': 0.015,
                'velocity': 0.001,
                'accuracy': 0.002,
                'status': 'NORMAL'
            },
            'pwv': {
                'ztd': 2.4567,
                'zhd': 2.3456,
                'zwd': 0.1111,
                'pwv': 17.89,
                'temperature': 25.0,
                'pressure': 1013.25,
                'humidity': 65.0,
                'atmospheric_condition': 'MODERATE'
            },
            'rainfall': {
                'predicted_rainfall': 5.67,
                'confidence': 0.82,
                'pwv': 17.89
            }
        }
    }

    try:
        output_path = "output/monitoring_report.pdf"
        os.makedirs("output", exist_ok=True)

        print(f"\n正在生成PDF报告...")
        report = gnss.GNSSMonitoringReport(output_path)
        pdf_file = report.create_comprehensive_report(test_results)

        print(f"✅ PDF报告已生成: {pdf_file}")
        print(f"  文件大小: {os.path.getsize(pdf_file) / 1024:.2f} KB")

    except Exception as e:
        print(f"⚠️  PDF生成失败: {e}")
        print("  请安装: pip install reportlab")


def demo_system_monitoring():
    """演示系统监控"""
    print("\n" + "=" * 80)
    print("4. System Monitoring Demo")
    print("=" * 80)

    print("\n正在收集系统指标...")

    # 获取监控服务
    monitoring = gnss.get_monitoring_service()

    # 收集指标
    monitoring.collect_metrics()

    # 获取摘要
    summary = monitoring.get_metrics_summary()

    print(f"\n系统资源使用情况:")
    print(f"  CPU使用率: {summary.get('cpu_usage_percent', 0):.1f}%")
    print(f"  内存使用率: {summary.get('memory_usage_percent', 0):.1f}%")
    print(f"  磁盘使用率: {summary.get('disk_usage_percent', 0):.1f}%")
    print(f"  运行时间: {summary.get('uptime_seconds', 0):.0f} 秒")

    # 健康检查
    print("\n执行健康检查...")
    health = monitoring.health_checker.run_checks()

    print(f"  系统状态: {health['status'].upper()}")
    for check_name, check_result in health['checks'].items():
        status = "✅" if check_result['healthy'] else "❌"
        print(f"    {status} {check_name}: {check_result['status']}")

    # Prometheus指标格式
    print("\nPrometheus格式指标示例:")
    prometheus_metrics = monitoring.get_prometheus_metrics()
    # 只显示前5行
    lines = prometheus_metrics.split('\n')[:5]
    for line in lines:
        if line:
            print(f"  {line}")
    print(f"  ... ({len(prometheus_metrics.split(chr(10)))} total lines)")


def demo_authentication():
    """演示用户认证"""
    print("\n" + "=" * 80)
    print("5. User Authentication Demo")
    print("=" * 80)

    try:
        # 创建认证服务
        auth_service = gnss.create_auth_service()

        print("\n用户认证功能:")
        print("  - JWT Token认证")
        print("  - 角色和权限管理")
        print("  - 密码加密")

        print("\n可用角色:")
        print(f"  - ADMIN: 完全访问权限")
        print(f"  - OPERATOR: 操作和分析权限")
        print(f"  - VIEWER: 只读权限")
        print(f"  - API_CLIENT: API访问权限")

        # 登录示例
        print("\n登录测试 (默认管理员账户):")
        print("  用户名: admin")
        print("  密码: admin123")

        token = auth_service.login("admin", "admin123")
        print(f"\n✅ 登录成功!")
        print(f"  Access Token: {token.access_token[:50]}...")
        print(f"  Token类型: {token.token_type}")
        print(f"  有效期: {token.expires_in}秒")

        # 验证token
        token_data = auth_service.verify_token(token.access_token)
        print(f"\nToken验证:")
        print(f"  用户: {token_data.username}")
        print(f"  角色: {', '.join(token_data.roles)}")
        print(f"  权限数量: {len(token_data.permissions)}")

    except Exception as e:
        print(f"⚠️  认证功能需要额外依赖: {e}")
        print("  安装: pip install passlib[bcrypt] python-jose[cryptography]")


def demo_api_server():
    """演示API服务器"""
    print("\n" + "=" * 80)
    print("6. RESTful API Server Demo")
    print("=" * 80)

    print("\nFastAPI RESTful API功能:")
    print("  - 完整的HTTP API接口")
    print("  - OpenAPI (Swagger) 文档")
    print("  - WebSocket实时通信")
    print("  - 自动数据验证")

    print("\n主要API端点:")
    print("  GET  /                    - API信息")
    print("  GET  /health              - 健康检查")
    print("  GET  /status              - 系统状态")
    print("  POST /analysis/run        - 运行分析")
    print("  GET  /data/gnss-ir        - 查询GNSS-IR数据")
    print("  GET  /data/deformation    - 查询形变数据")
    print("  GET  /data/pwv            - 查询PWV数据")
    print("  GET  /alerts              - 查询告警")
    print("  GET  /statistics/overview - 系统统计")

    print("\n启动API服务器:")
    print("""
    # 方法1: 使用便捷函数
    from gnss_monitoring import run_api_server
    run_api_server(host="0.0.0.0", port=8000)

    # 方法2: 使用uvicorn
    uvicorn gnss_monitoring.api:app --host 0.0.0.0 --port 8000

    # 访问API文档: http://localhost:8000/docs
    """)

    print("\n⚠️  需要启动服务器后才能访问API")
    print("  文档地址: http://localhost:8000/docs")


def demo_advanced_visualization():
    """演示高级可视化"""
    print("\n" + "=" * 80)
    print("7. Advanced Visualization Demo")
    print("=" * 80)

    print("\n高级可视化功能:")
    print("  - 3D轨迹和形变可视化")
    print("  - 交互式时间序列图")
    print("  - 热力图和相关性分析")
    print("  - 多面板仪表板")

    try:
        # 3D轨迹示例
        print("\n3D轨迹可视化示例:")
        coordinates = [
            (0, 0, 0),
            (1.2, 0.5, 0.1),
            (2.1, 1.2, 0.05),
            (3.0, 1.8, -0.02),
            (3.8, 2.5, -0.08)
        ]

        print(f"  生成3D轨迹图 ({len(coordinates)}个点)")
        print("  需要Plotly支持")

        print("\n使用示例:")
        print("""
        from gnss_monitoring import Advanced3DVisualizer, save_plotly_figure

        visualizer = Advanced3DVisualizer()
        fig = visualizer.plot_3d_trajectory(coordinates, title="Station Trajectory")
        save_plotly_figure(fig, "trajectory_3d.html")
        """)

    except Exception as e:
        print(f"⚠️  需要Plotly: {e}")
        print("  安装: pip install plotly kaleido")


def main():
    """主函数"""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "CHS-BDS Phase 4企业级功能演示" + " " * 28 + "║")
    print("╚" + "═" * 78 + "╝")

    print(f"\n系统版本: v{gnss.__version__}")
    print(f"日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 功能可用性检查
    print("\n功能可用性检查:")
    print(f"  ✅ Database: DatabaseManager 可用")
    print(f"  ✅ RINEX Parser: RINEXParser 可用")
    print(f"  ✅ Monitoring: MonitoringService 可用")
    print(f"  {'✅' if gnss.API_AVAILABLE else '❌'} API: FastAPI {'可用' if gnss.API_AVAILABLE else '不可用'}")
    print(f"  {'✅' if gnss.PDF_AVAILABLE else '❌'} PDF: ReportLab {'可用' if gnss.PDF_AVAILABLE else '不可用'}")
    print(f"  {'✅' if gnss.WEBSOCKET_AVAILABLE else '❌'} WebSocket: {'可用' if gnss.WEBSOCKET_AVAILABLE else '不可用'}")
    print(f"  {'✅' if gnss.AUTH_AVAILABLE else '❌'} Authentication: {'可用' if gnss.AUTH_AVAILABLE else '不可用'}")
    print(f"  {'✅' if gnss.ADVANCED_VIZ_AVAILABLE else '❌'} Advanced Viz: {'可用' if gnss.ADVANCED_VIZ_AVAILABLE else '不可用'}")

    # 运行各功能演示
    try:
        demo_database_integration()
        demo_rinex_parser()
        demo_pdf_report_generation()
        demo_system_monitoring()
        demo_authentication()
        demo_api_server()
        demo_advanced_visualization()

    except KeyboardInterrupt:
        print("\n\n⚠️  演示已中断")
    except Exception as e:
        print(f"\n\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("演示完成!")
    print("=" * 80)

    print("\n下一步:")
    print("  1. 安装所有依赖: pip install -r requirements.txt")
    print("  2. 配置PostgreSQL数据库")
    print("  3. 启动API服务器: uvicorn gnss_monitoring.api:app")
    print("  4. 访问API文档: http://localhost:8000/docs")
    print("  5. 查看完整文档: DEVELOPMENT_PHASE4.md")


if __name__ == "__main__":
    main()
