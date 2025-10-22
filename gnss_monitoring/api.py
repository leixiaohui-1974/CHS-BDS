"""
RESTful API服务模块 - FastAPI实现

本模块提供完整的HTTP API服务，包括：
- GNSS监测数据查询和管理
- 实时分析任务触发
- 告警管理
- 系统状态监控
- 统计信息查询
- WebSocket实时推送支持
- OpenAPI文档

Author: Lei Xiaohui
Date: 2025-01-22
"""

import os
import json
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum

try:
    from fastapi import FastAPI, HTTPException, Depends, Query, Path, Body, status
    from fastapi.responses import JSONResponse, StreamingResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field, validator
    from uvicorn import run as uvicorn_run
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    # 创建占位符类
    class BaseModel:
        pass

from .logger import get_logger
from .exceptions import CHSBDSException
from .main import CHSBDSSystem
from .database import DatabaseManager, DatabaseConfig
from .alerts import AlertManager, AlertLevel

logger = get_logger(__name__)


class APIError(CHSBDSException):
    """API异常"""
    pass


# ==================== Pydantic Models ====================

class AnalysisModule(str, Enum):
    """分析模块枚举"""
    GNSS_IR = "gnss_ir"
    DEFORMATION = "deformation"
    PWV = "pwv"
    RAINFALL = "rainfall"
    ALL = "all"


class AlertLevelEnum(str, Enum):
    """告警级别枚举"""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class TimeRange(BaseModel):
    """时间范围"""
    start_time: Optional[datetime] = Field(None, description="起始时间")
    end_time: Optional[datetime] = Field(None, description="结束时间")

    @validator('end_time')
    def validate_time_range(cls, v, values):
        if v and 'start_time' in values and values['start_time']:
            if v < values['start_time']:
                raise ValueError('end_time must be after start_time')
        return v


class StationQuery(BaseModel):
    """测站查询参数"""
    station_id: Optional[str] = Field(None, description="测站ID")
    time_range: Optional[TimeRange] = None
    limit: int = Field(1000, ge=1, le=10000, description="返回记录数限制")


class AnalysisRequest(BaseModel):
    """分析请求"""
    module: AnalysisModule = Field(..., description="分析模块")
    parameters: Optional[Dict[str, Any]] = Field(None, description="模块参数")
    save_to_db: bool = Field(True, description="是否保存到数据库")


class AnalysisResponse(BaseModel):
    """分析响应"""
    task_id: str = Field(..., description="任务ID")
    module: str = Field(..., description="分析模块")
    status: str = Field(..., description="任务状态")
    started_at: datetime = Field(..., description="开始时间")
    results: Optional[Dict[str, Any]] = Field(None, description="分析结果")
    error: Optional[str] = Field(None, description="错误信息")


class AlertResponse(BaseModel):
    """告警响应"""
    id: int
    alert_id: str
    title: str
    message: str
    level: str
    module: str
    timestamp: datetime
    acknowledged: bool


class SystemStatus(BaseModel):
    """系统状态"""
    status: str = Field(..., description="系统状态")
    uptime: float = Field(..., description="运行时间（秒）")
    version: str = Field(..., description="系统版本")
    active_tasks: int = Field(..., description="活动任务数")
    database_connected: bool = Field(..., description="数据库连接状态")
    total_alerts: int = Field(..., description="总告警数")
    unacknowledged_alerts: int = Field(..., description="未确认告警数")


class StatisticsResponse(BaseModel):
    """统计信息响应"""
    table_name: str
    total_records: int
    earliest_record: Optional[datetime]
    latest_record: Optional[datetime]


# ==================== FastAPI Application ====================

class CHSBDSAPI:
    """CHS-BDS RESTful API服务"""

    def __init__(
        self,
        title: str = "CHS-BDS GNSS Monitoring API",
        version: str = "0.4.0",
        enable_database: bool = True,
        enable_cors: bool = True
    ):
        if not FASTAPI_AVAILABLE:
            raise APIError("FastAPI is not installed. Install with: pip install fastapi uvicorn")

        self.app = FastAPI(
            title=title,
            version=version,
            description="北斗/GPS多模GNSS综合监测系统 RESTful API",
            docs_url="/docs",
            redoc_url="/redoc",
            openapi_url="/openapi.json"
        )

        # 启动时间
        self.start_time = datetime.now()

        # CHS-BDS系统实例
        self.system = CHSBDSSystem()

        # 数据库管理器
        self.db_manager = None
        if enable_database:
            try:
                self.db_manager = DatabaseManager()
                self.db_manager.initialize()
                logger.info("Database initialized for API")
            except Exception as e:
                logger.warning(f"Failed to initialize database: {e}")

        # 告警管理器
        self.alert_manager = AlertManager()

        # 活动任务跟踪
        self.active_tasks: Dict[str, Dict[str, Any]] = {}

        # 配置CORS
        if enable_cors:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],  # 生产环境应限制具体域名
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

        # 注册路由
        self._register_routes()

    def _register_routes(self):
        """注册API路由"""

        # ==================== 健康检查和状态 ====================

        @self.app.get("/", tags=["System"])
        async def root():
            """API根路径"""
            return {
                "name": "CHS-BDS GNSS Monitoring API",
                "version": "0.4.0",
                "status": "running",
                "documentation": "/docs"
            }

        @self.app.get("/health", tags=["System"])
        async def health_check():
            """健康检查"""
            db_status = False
            if self.db_manager:
                try:
                    stats = self.db_manager.get_statistics("gnss_ir_data")
                    db_status = True
                except:
                    pass

            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "database": "connected" if db_status else "disconnected"
            }

        @self.app.get("/status", response_model=SystemStatus, tags=["System"])
        async def get_status():
            """获取系统状态"""
            uptime = (datetime.now() - self.start_time).total_seconds()

            # 获取告警统计
            total_alerts = 0
            unacknowledged_alerts = 0
            if self.db_manager:
                try:
                    all_alerts = self.db_manager.query_alerts(limit=10000)
                    total_alerts = len(all_alerts)
                    unacknowledged_alerts = len([a for a in all_alerts if not a.get('acknowledged')])
                except:
                    pass

            return SystemStatus(
                status="running",
                uptime=uptime,
                version="0.4.0",
                active_tasks=len(self.active_tasks),
                database_connected=self.db_manager is not None,
                total_alerts=total_alerts,
                unacknowledged_alerts=unacknowledged_alerts
            )

        # ==================== 分析任务 ====================

        @self.app.post("/analysis/run", response_model=AnalysisResponse, tags=["Analysis"])
        async def run_analysis(request: AnalysisRequest):
            """触发分析任务"""
            import uuid

            task_id = str(uuid.uuid4())
            started_at = datetime.now()

            try:
                # 运行分析
                if request.module == AnalysisModule.ALL:
                    results = self.system.run_all()
                elif request.module == AnalysisModule.GNSS_IR:
                    results = self.system.run_gnss_ir()
                elif request.module == AnalysisModule.DEFORMATION:
                    results = self.system.run_deformation()
                elif request.module == AnalysisModule.PWV:
                    results = self.system.run_pwv_estimation()
                elif request.module == AnalysisModule.RAINFALL:
                    results = self.system.run_rainfall_prediction()
                else:
                    raise HTTPException(status_code=400, detail=f"Unknown module: {request.module}")

                # 保存到数据库（如果启用）
                if request.save_to_db and self.db_manager:
                    self._save_results_to_db(request.module.value, results)

                return AnalysisResponse(
                    task_id=task_id,
                    module=request.module.value,
                    status="completed",
                    started_at=started_at,
                    results=results,
                    error=None
                )

            except Exception as e:
                logger.error(f"Analysis failed: {e}")
                return AnalysisResponse(
                    task_id=task_id,
                    module=request.module.value,
                    status="failed",
                    started_at=started_at,
                    results=None,
                    error=str(e)
                )

        # ==================== 数据查询 ====================

        @self.app.get("/data/gnss-ir", tags=["Data"])
        async def query_gnss_ir_data(
            station_id: Optional[str] = Query(None, description="测站ID"),
            start_time: Optional[datetime] = Query(None, description="起始时间"),
            end_time: Optional[datetime] = Query(None, description="结束时间"),
            limit: int = Query(1000, ge=1, le=10000, description="返回记录数")
        ):
            """查询GNSS-IR数据"""
            if not self.db_manager:
                raise HTTPException(status_code=503, detail="Database not available")

            try:
                data = self.db_manager.query_gnss_ir_data(
                    station_id=station_id,
                    start_time=start_time,
                    end_time=end_time,
                    limit=limit
                )
                return {"count": len(data), "data": data}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/data/deformation", tags=["Data"])
        async def query_deformation_data(
            station_id: Optional[str] = Query(None, description="测站ID"),
            start_time: Optional[datetime] = Query(None, description="起始时间"),
            end_time: Optional[datetime] = Query(None, description="结束时间"),
            limit: int = Query(1000, ge=1, le=10000, description="返回记录数")
        ):
            """查询形变监测数据"""
            if not self.db_manager:
                raise HTTPException(status_code=503, detail="Database not available")

            try:
                data = self.db_manager.query_deformation_data(
                    station_id=station_id,
                    start_time=start_time,
                    end_time=end_time,
                    limit=limit
                )
                return {"count": len(data), "data": data}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/data/pwv", tags=["Data"])
        async def query_pwv_data(
            station_id: Optional[str] = Query(None, description="测站ID"),
            start_time: Optional[datetime] = Query(None, description="起始时间"),
            end_time: Optional[datetime] = Query(None, description="结束时间"),
            limit: int = Query(1000, ge=1, le=10000, description="返回记录数")
        ):
            """查询PWV数据"""
            if not self.db_manager:
                raise HTTPException(status_code=503, detail="Database not available")

            try:
                data = self.db_manager.query_pwv_data(
                    station_id=station_id,
                    start_time=start_time,
                    end_time=end_time,
                    limit=limit
                )
                return {"count": len(data), "data": data}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        # ==================== 告警管理 ====================

        @self.app.get("/alerts", tags=["Alerts"])
        async def query_alerts(
            level: Optional[AlertLevelEnum] = Query(None, description="告警级别"),
            module: Optional[str] = Query(None, description="模块名称"),
            acknowledged: Optional[bool] = Query(None, description="是否已确认"),
            limit: int = Query(100, ge=1, le=1000, description="返回记录数")
        ):
            """查询告警记录"""
            if not self.db_manager:
                raise HTTPException(status_code=503, detail="Database not available")

            try:
                alerts = self.db_manager.query_alerts(
                    level=level.value if level else None,
                    module=module,
                    acknowledged=acknowledged,
                    limit=limit
                )
                return {"count": len(alerts), "alerts": alerts}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/alerts/{alert_id}/acknowledge", tags=["Alerts"])
        async def acknowledge_alert(
            alert_id: int = Path(..., description="告警ID"),
            acknowledged_by: str = Body(..., embed=True, description="确认人")
        ):
            """确认告警"""
            if not self.db_manager:
                raise HTTPException(status_code=503, detail="Database not available")

            try:
                self.db_manager.acknowledge_alert(alert_id, acknowledged_by)
                return {"message": f"Alert {alert_id} acknowledged by {acknowledged_by}"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        # ==================== 统计信息 ====================

        @self.app.get("/statistics/{table_name}", response_model=StatisticsResponse, tags=["Statistics"])
        async def get_statistics(
            table_name: str = Path(..., description="表名",
                                   regex="^(gnss_ir_data|deformation_data|pwv_data|rainfall_data)$")
        ):
            """获取表统计信息"""
            if not self.db_manager:
                raise HTTPException(status_code=503, detail="Database not available")

            try:
                stats = self.db_manager.get_statistics(table_name)
                return StatisticsResponse(
                    table_name=table_name,
                    total_records=stats.get('total_records', 0),
                    earliest_record=stats.get('earliest_record'),
                    latest_record=stats.get('latest_record')
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/statistics/overview", tags=["Statistics"])
        async def get_overview_statistics():
            """获取系统概览统计"""
            if not self.db_manager:
                raise HTTPException(status_code=503, detail="Database not available")

            try:
                tables = ["gnss_ir_data", "deformation_data", "pwv_data", "rainfall_data"]
                overview = {}

                for table in tables:
                    stats = self.db_manager.get_statistics(table)
                    overview[table] = stats

                return overview
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        # ==================== 数据管理 ====================

        @self.app.delete("/data/cleanup/{table_name}", tags=["Management"])
        async def cleanup_old_data(
            table_name: str = Path(..., description="表名"),
            days: int = Query(90, ge=1, le=365, description="保留天数")
        ):
            """清理旧数据"""
            if not self.db_manager:
                raise HTTPException(status_code=503, detail="Database not available")

            try:
                deleted_count = self.db_manager.cleanup_old_data(table_name, days)
                return {
                    "table_name": table_name,
                    "deleted_records": deleted_count,
                    "retention_days": days
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

    def _save_results_to_db(self, module: str, results: Dict[str, Any]):
        """保存分析结果到数据库"""
        if not self.db_manager:
            return

        try:
            timestamp = datetime.now()
            station_id = "TEST_STATION"  # 应该从配置或参数中获取

            if module == "gnss_ir":
                # 保存GNSS-IR结果
                if 'water_level' in results:
                    data = [{
                        'timestamp': timestamp,
                        'station_id': station_id,
                        'satellite_id': None,
                        'elevation_angle': None,
                        'azimuth_angle': None,
                        'snr_value': None,
                        'water_level': results['water_level'],
                        'quality_score': results.get('confidence', 0.0),
                        'metadata': json.dumps(results)
                    }]
                    self.db_manager.insert_gnss_ir_data(data)

            elif module == "deformation":
                # 保存形变监测结果
                if 'displacement' in results:
                    data = [{
                        'timestamp': timestamp,
                        'station_id': station_id,
                        'baseline_id': 'BASELINE_01',
                        'dx': results.get('dx', 0.0),
                        'dy': results.get('dy', 0.0),
                        'dz': results.get('dz', 0.0),
                        'displacement': results['displacement'],
                        'velocity': results.get('velocity', 0.0),
                        'accuracy': results.get('accuracy', 0.0),
                        'metadata': json.dumps(results)
                    }]
                    self.db_manager.insert_deformation_data(data)

            elif module == "pwv":
                # 保存PWV结果
                if 'pwv' in results:
                    data = [{
                        'timestamp': timestamp,
                        'station_id': station_id,
                        'ztd': results.get('ztd', 0.0),
                        'zhd': results.get('zhd', 0.0),
                        'zwd': results.get('zwd', 0.0),
                        'pwv': results['pwv'],
                        'temperature': results.get('temperature', 20.0),
                        'pressure': results.get('pressure', 1013.25),
                        'humidity': results.get('humidity', 50.0),
                        'metadata': json.dumps(results)
                    }]
                    self.db_manager.insert_pwv_data(data)

            elif module == "rainfall":
                # 保存降雨预测结果
                if 'predicted_rainfall' in results:
                    data = [{
                        'timestamp': timestamp,
                        'station_id': station_id,
                        'pwv': results.get('pwv', 0.0),
                        'rainfall_predicted': results['predicted_rainfall'],
                        'rainfall_actual': None,
                        'confidence': results.get('confidence', 0.0),
                        'model_version': 'v1.0',
                        'metadata': json.dumps(results)
                    }]
                    self.db_manager.insert_rainfall_data(data)

        except Exception as e:
            logger.error(f"Failed to save results to database: {e}")

    def run(self, host: str = "0.0.0.0", port: int = 8000, **kwargs):
        """启动API服务"""
        logger.info(f"Starting CHS-BDS API server on {host}:{port}")
        uvicorn_run(self.app, host=host, port=port, **kwargs)


# ==================== 便捷函数 ====================

def create_api(
    enable_database: bool = True,
    enable_cors: bool = True
) -> CHSBDSAPI:
    """创建API实例"""
    return CHSBDSAPI(
        enable_database=enable_database,
        enable_cors=enable_cors
    )


def run_api_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    enable_database: bool = True,
    **kwargs
):
    """运行API服务器"""
    api = create_api(enable_database=enable_database)
    api.run(host=host, port=port, **kwargs)


if __name__ == "__main__":
    # 测试代码
    print("CHS-BDS API module")

    if not FASTAPI_AVAILABLE:
        print("Warning: FastAPI is not installed")
        print("Install with: pip install fastapi uvicorn")
    else:
        print("FastAPI is available")
        print("\nTo run the API server:")
        print("  python -m gnss_monitoring.api")
        print("  OR")
        print("  uvicorn gnss_monitoring.api:app --reload")

        # 创建API实例用于测试
        try:
            api = create_api(enable_database=False)
            print(f"\nAPI created successfully")
            print(f"Routes: {len(api.app.routes)}")

            # 运行服务器
            # api.run(host="0.0.0.0", port=8000)
        except Exception as e:
            print(f"Failed to create API: {e}")
