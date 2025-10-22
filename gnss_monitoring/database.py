"""
数据库集成模块 - PostgreSQL/TimescaleDB支持

本模块提供时间序列数据的持久化存储，支持：
- PostgreSQL关系型数据库
- TimescaleDB时间序列扩展
- 自动表创建和schema管理
- 批量数据插入
- 高效查询接口
- 数据归档和清理

Author: Lei Xiaohui
Date: 2025-01-22
"""

import os
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from contextlib import contextmanager
import logging

try:
    import psycopg2
    from psycopg2 import pool, sql
    from psycopg2.extras import execute_batch, RealDictCursor
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

try:
    from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, JSON, Text
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker, Session
    from sqlalchemy.pool import QueuePool
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

from .logger import get_logger
from .exceptions import CHSBDSException

logger = get_logger(__name__)

Base = declarative_base() if SQLALCHEMY_AVAILABLE else None


class DatabaseError(CHSBDSException):
    """数据库操作异常"""
    pass


class DatabaseConfig:
    """数据库配置"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "chs_bds",
        user: str = "postgres",
        password: str = "",
        min_connections: int = 2,
        max_connections: int = 10,
        enable_timescaledb: bool = True
    ):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.enable_timescaledb = enable_timescaledb

    @classmethod
    def from_env(cls) -> 'DatabaseConfig':
        """从环境变量加载配置"""
        return cls(
            host=os.getenv('DB_HOST', 'localhost'),
            port=int(os.getenv('DB_PORT', '5432')),
            database=os.getenv('DB_NAME', 'chs_bds'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', ''),
            min_connections=int(os.getenv('DB_MIN_CONN', '2')),
            max_connections=int(os.getenv('DB_MAX_CONN', '10')),
            enable_timescaledb=os.getenv('DB_TIMESCALE', 'true').lower() == 'true'
        )

    def get_connection_string(self) -> str:
        """获取数据库连接字符串"""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


class DatabaseManager:
    """数据库管理器 - 使用psycopg2连接池"""

    def __init__(self, config: Optional[DatabaseConfig] = None):
        if not PSYCOPG2_AVAILABLE:
            raise DatabaseError("psycopg2 is not installed. Install with: pip install psycopg2-binary")

        self.config = config or DatabaseConfig.from_env()
        self.connection_pool = None
        self._initialized = False

    def initialize(self):
        """初始化数据库连接池"""
        if self._initialized:
            return

        try:
            self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=self.config.min_connections,
                maxconn=self.config.max_connections,
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.user,
                password=self.config.password
            )
            logger.info(f"Database connection pool created: {self.config.host}:{self.config.port}/{self.config.database}")

            # 创建schema和表
            self._create_schema()
            self._initialized = True

        except Exception as e:
            raise DatabaseError(f"Failed to initialize database: {e}")

    @contextmanager
    def get_connection(self):
        """获取数据库连接（上下文管理器）"""
        if not self._initialized:
            self.initialize()

        conn = None
        try:
            conn = self.connection_pool.getconn()
            yield conn
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            raise DatabaseError(f"Database operation failed: {e}")
        finally:
            if conn:
                self.connection_pool.putconn(conn)

    def _create_schema(self):
        """创建数据库schema和表"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # 启用TimescaleDB扩展（如果配置了）
            if self.config.enable_timescaledb:
                try:
                    cursor.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")
                    logger.info("TimescaleDB extension enabled")
                except Exception as e:
                    logger.warning(f"Failed to enable TimescaleDB: {e}")

            # 创建GNSS-IR数据表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS gnss_ir_data (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL,
                    station_id VARCHAR(50) NOT NULL,
                    satellite_id VARCHAR(20),
                    elevation_angle FLOAT,
                    azimuth_angle FLOAT,
                    snr_value FLOAT,
                    water_level FLOAT,
                    quality_score FLOAT,
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)

            # 创建形变监测数据表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS deformation_data (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL,
                    station_id VARCHAR(50) NOT NULL,
                    baseline_id VARCHAR(50),
                    dx FLOAT,
                    dy FLOAT,
                    dz FLOAT,
                    displacement FLOAT,
                    velocity FLOAT,
                    accuracy FLOAT,
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)

            # 创建PWV估算数据表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pwv_data (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL,
                    station_id VARCHAR(50) NOT NULL,
                    ztd FLOAT,
                    zhd FLOAT,
                    zwd FLOAT,
                    pwv FLOAT,
                    temperature FLOAT,
                    pressure FLOAT,
                    humidity FLOAT,
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)

            # 创建降雨预测数据表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS rainfall_data (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL,
                    station_id VARCHAR(50) NOT NULL,
                    pwv FLOAT,
                    rainfall_predicted FLOAT,
                    rainfall_actual FLOAT,
                    confidence FLOAT,
                    model_version VARCHAR(50),
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)

            # 创建质量控制记录表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS quality_control_records (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL,
                    module VARCHAR(50) NOT NULL,
                    data_source VARCHAR(200),
                    completeness_score FLOAT,
                    outlier_count INTEGER,
                    quality_score FLOAT,
                    status VARCHAR(20),
                    report_path TEXT,
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)

            # 创建告警记录表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alert_records (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL,
                    alert_id VARCHAR(100) NOT NULL,
                    title VARCHAR(200),
                    message TEXT,
                    level VARCHAR(20),
                    module VARCHAR(50),
                    status VARCHAR(20),
                    acknowledged BOOLEAN DEFAULT FALSE,
                    acknowledged_by VARCHAR(100),
                    acknowledged_at TIMESTAMPTZ,
                    resolved_at TIMESTAMPTZ,
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)

            # 创建系统监控指标表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL,
                    metric_name VARCHAR(100) NOT NULL,
                    metric_value FLOAT,
                    metric_unit VARCHAR(20),
                    labels JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)

            # 创建索引以提高查询性能
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_gnss_ir_timestamp ON gnss_ir_data(timestamp DESC);",
                "CREATE INDEX IF NOT EXISTS idx_gnss_ir_station ON gnss_ir_data(station_id);",
                "CREATE INDEX IF NOT EXISTS idx_deformation_timestamp ON deformation_data(timestamp DESC);",
                "CREATE INDEX IF NOT EXISTS idx_deformation_station ON deformation_data(station_id);",
                "CREATE INDEX IF NOT EXISTS idx_pwv_timestamp ON pwv_data(timestamp DESC);",
                "CREATE INDEX IF NOT EXISTS idx_pwv_station ON pwv_data(station_id);",
                "CREATE INDEX IF NOT EXISTS idx_rainfall_timestamp ON rainfall_data(timestamp DESC);",
                "CREATE INDEX IF NOT EXISTS idx_quality_timestamp ON quality_control_records(timestamp DESC);",
                "CREATE INDEX IF NOT EXISTS idx_alert_timestamp ON alert_records(timestamp DESC);",
                "CREATE INDEX IF NOT EXISTS idx_alert_level ON alert_records(level);",
                "CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON system_metrics(timestamp DESC);",
                "CREATE INDEX IF NOT EXISTS idx_metrics_name ON system_metrics(metric_name);",
            ]

            for index_sql in indexes:
                cursor.execute(index_sql)

            # 如果启用了TimescaleDB，转换为hypertable
            if self.config.enable_timescaledb:
                hypertables = [
                    ("gnss_ir_data", "timestamp"),
                    ("deformation_data", "timestamp"),
                    ("pwv_data", "timestamp"),
                    ("rainfall_data", "timestamp"),
                    ("quality_control_records", "timestamp"),
                    ("alert_records", "timestamp"),
                    ("system_metrics", "timestamp"),
                ]

                for table_name, time_column in hypertables:
                    try:
                        cursor.execute(
                            f"SELECT create_hypertable('{table_name}', '{time_column}', "
                            f"if_not_exists => TRUE);"
                        )
                        logger.info(f"Created hypertable: {table_name}")
                    except Exception as e:
                        logger.debug(f"Hypertable {table_name} may already exist: {e}")

            cursor.close()
            logger.info("Database schema created successfully")

    def insert_gnss_ir_data(self, data: List[Dict[str, Any]]):
        """批量插入GNSS-IR数据"""
        if not data:
            return

        with self.get_connection() as conn:
            cursor = conn.cursor()

            insert_query = """
                INSERT INTO gnss_ir_data
                (timestamp, station_id, satellite_id, elevation_angle, azimuth_angle,
                 snr_value, water_level, quality_score, metadata)
                VALUES (%(timestamp)s, %(station_id)s, %(satellite_id)s, %(elevation_angle)s,
                        %(azimuth_angle)s, %(snr_value)s, %(water_level)s,
                        %(quality_score)s, %(metadata)s)
            """

            execute_batch(cursor, insert_query, data)
            cursor.close()
            logger.info(f"Inserted {len(data)} GNSS-IR records")

    def insert_deformation_data(self, data: List[Dict[str, Any]]):
        """批量插入形变监测数据"""
        if not data:
            return

        with self.get_connection() as conn:
            cursor = conn.cursor()

            insert_query = """
                INSERT INTO deformation_data
                (timestamp, station_id, baseline_id, dx, dy, dz, displacement,
                 velocity, accuracy, metadata)
                VALUES (%(timestamp)s, %(station_id)s, %(baseline_id)s, %(dx)s, %(dy)s,
                        %(dz)s, %(displacement)s, %(velocity)s, %(accuracy)s, %(metadata)s)
            """

            execute_batch(cursor, insert_query, data)
            cursor.close()
            logger.info(f"Inserted {len(data)} deformation records")

    def insert_pwv_data(self, data: List[Dict[str, Any]]):
        """批量插入PWV数据"""
        if not data:
            return

        with self.get_connection() as conn:
            cursor = conn.cursor()

            insert_query = """
                INSERT INTO pwv_data
                (timestamp, station_id, ztd, zhd, zwd, pwv, temperature,
                 pressure, humidity, metadata)
                VALUES (%(timestamp)s, %(station_id)s, %(ztd)s, %(zhd)s, %(zwd)s,
                        %(pwv)s, %(temperature)s, %(pressure)s, %(humidity)s, %(metadata)s)
            """

            execute_batch(cursor, insert_query, data)
            cursor.close()
            logger.info(f"Inserted {len(data)} PWV records")

    def insert_rainfall_data(self, data: List[Dict[str, Any]]):
        """批量插入降雨预测数据"""
        if not data:
            return

        with self.get_connection() as conn:
            cursor = conn.cursor()

            insert_query = """
                INSERT INTO rainfall_data
                (timestamp, station_id, pwv, rainfall_predicted, rainfall_actual,
                 confidence, model_version, metadata)
                VALUES (%(timestamp)s, %(station_id)s, %(pwv)s, %(rainfall_predicted)s,
                        %(rainfall_actual)s, %(confidence)s, %(model_version)s, %(metadata)s)
            """

            execute_batch(cursor, insert_query, data)
            cursor.close()
            logger.info(f"Inserted {len(data)} rainfall records")

    def insert_quality_record(self, record: Dict[str, Any]):
        """插入质量控制记录"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO quality_control_records
                (timestamp, module, data_source, completeness_score, outlier_count,
                 quality_score, status, report_path, metadata)
                VALUES (%(timestamp)s, %(module)s, %(data_source)s, %(completeness_score)s,
                        %(outlier_count)s, %(quality_score)s, %(status)s, %(report_path)s,
                        %(metadata)s)
            """, record)

            cursor.close()

    def insert_alert(self, alert: Dict[str, Any]):
        """插入告警记录"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO alert_records
                (timestamp, alert_id, title, message, level, module, status, metadata)
                VALUES (%(timestamp)s, %(alert_id)s, %(title)s, %(message)s, %(level)s,
                        %(module)s, %(status)s, %(metadata)s)
                RETURNING id
            """, alert)

            alert_id = cursor.fetchone()[0]
            cursor.close()
            return alert_id

    def insert_metric(self, metric: Dict[str, Any]):
        """插入系统监控指标"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO system_metrics
                (timestamp, metric_name, metric_value, metric_unit, labels)
                VALUES (%(timestamp)s, %(metric_name)s, %(metric_value)s,
                        %(metric_unit)s, %(labels)s)
            """, metric)

            cursor.close()

    def query_gnss_ir_data(
        self,
        station_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """查询GNSS-IR数据"""
        with self.get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            conditions = []
            params = {}

            if station_id:
                conditions.append("station_id = %(station_id)s")
                params['station_id'] = station_id

            if start_time:
                conditions.append("timestamp >= %(start_time)s")
                params['start_time'] = start_time

            if end_time:
                conditions.append("timestamp <= %(end_time)s")
                params['end_time'] = end_time

            where_clause = " AND ".join(conditions) if conditions else "TRUE"
            params['limit'] = limit

            query = f"""
                SELECT * FROM gnss_ir_data
                WHERE {where_clause}
                ORDER BY timestamp DESC
                LIMIT %(limit)s
            """

            cursor.execute(query, params)
            results = cursor.fetchall()
            cursor.close()

            return [dict(row) for row in results]

    def query_deformation_data(
        self,
        station_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """查询形变监测数据"""
        with self.get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            conditions = []
            params = {}

            if station_id:
                conditions.append("station_id = %(station_id)s")
                params['station_id'] = station_id

            if start_time:
                conditions.append("timestamp >= %(start_time)s")
                params['start_time'] = start_time

            if end_time:
                conditions.append("timestamp <= %(end_time)s")
                params['end_time'] = end_time

            where_clause = " AND ".join(conditions) if conditions else "TRUE"
            params['limit'] = limit

            query = f"""
                SELECT * FROM deformation_data
                WHERE {where_clause}
                ORDER BY timestamp DESC
                LIMIT %(limit)s
            """

            cursor.execute(query, params)
            results = cursor.fetchall()
            cursor.close()

            return [dict(row) for row in results]

    def query_pwv_data(
        self,
        station_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """查询PWV数据"""
        with self.get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            conditions = []
            params = {}

            if station_id:
                conditions.append("station_id = %(station_id)s")
                params['station_id'] = station_id

            if start_time:
                conditions.append("timestamp >= %(start_time)s")
                params['start_time'] = start_time

            if end_time:
                conditions.append("timestamp <= %(end_time)s")
                params['end_time'] = end_time

            where_clause = " AND ".join(conditions) if conditions else "TRUE"
            params['limit'] = limit

            query = f"""
                SELECT * FROM pwv_data
                WHERE {where_clause}
                ORDER BY timestamp DESC
                LIMIT %(limit)s
            """

            cursor.execute(query, params)
            results = cursor.fetchall()
            cursor.close()

            return [dict(row) for row in results]

    def query_alerts(
        self,
        level: Optional[str] = None,
        module: Optional[str] = None,
        start_time: Optional[datetime] = None,
        acknowledged: Optional[bool] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """查询告警记录"""
        with self.get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            conditions = []
            params = {}

            if level:
                conditions.append("level = %(level)s")
                params['level'] = level

            if module:
                conditions.append("module = %(module)s")
                params['module'] = module

            if start_time:
                conditions.append("timestamp >= %(start_time)s")
                params['start_time'] = start_time

            if acknowledged is not None:
                conditions.append("acknowledged = %(acknowledged)s")
                params['acknowledged'] = acknowledged

            where_clause = " AND ".join(conditions) if conditions else "TRUE"
            params['limit'] = limit

            query = f"""
                SELECT * FROM alert_records
                WHERE {where_clause}
                ORDER BY timestamp DESC
                LIMIT %(limit)s
            """

            cursor.execute(query, params)
            results = cursor.fetchall()
            cursor.close()

            return [dict(row) for row in results]

    def acknowledge_alert(self, alert_id: int, acknowledged_by: str):
        """确认告警"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE alert_records
                SET acknowledged = TRUE,
                    acknowledged_by = %s,
                    acknowledged_at = NOW()
                WHERE id = %s
            """, (acknowledged_by, alert_id))

            cursor.close()

    def get_statistics(self, table_name: str) -> Dict[str, Any]:
        """获取表统计信息"""
        with self.get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            cursor.execute(f"""
                SELECT
                    COUNT(*) as total_records,
                    MIN(timestamp) as earliest_record,
                    MAX(timestamp) as latest_record
                FROM {table_name}
            """)

            stats = dict(cursor.fetchone())
            cursor.close()

            return stats

    def cleanup_old_data(self, table_name: str, days: int = 90):
        """清理旧数据"""
        cutoff_date = datetime.now() - timedelta(days=days)

        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(f"""
                DELETE FROM {table_name}
                WHERE timestamp < %s
            """, (cutoff_date,))

            deleted_count = cursor.rowcount
            cursor.close()

            logger.info(f"Cleaned up {deleted_count} old records from {table_name}")
            return deleted_count

    def close(self):
        """关闭连接池"""
        if self.connection_pool:
            self.connection_pool.closeall()
            logger.info("Database connection pool closed")


# 便捷函数
def get_database_manager(config: Optional[DatabaseConfig] = None) -> DatabaseManager:
    """获取数据库管理器实例"""
    return DatabaseManager(config)


if __name__ == "__main__":
    # 测试代码
    print("Database module test")

    if not PSYCOPG2_AVAILABLE:
        print("Warning: psycopg2 is not installed")
        print("Install with: pip install psycopg2-binary")
    else:
        print("psycopg2 is available")

        # 示例：创建数据库管理器
        config = DatabaseConfig(
            host="localhost",
            database="chs_bds_test",
            user="postgres",
            password="your_password"
        )

        print(f"Connection string: {config.get_connection_string()}")
