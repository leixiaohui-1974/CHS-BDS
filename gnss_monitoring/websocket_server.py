"""
WebSocket实时通信模块

本模块提供WebSocket服务，用于实时数据推送：
- 实时监测数据推送
- 告警实时通知
- 系统状态更新
- 双向通信
- 连接管理

Author: Lei Xiaohui
Date: 2025-01-22
"""

import asyncio
import json
from typing import Dict, Set, Optional, Any, List
from datetime import datetime
from enum import Enum

try:
    from fastapi import WebSocket, WebSocketDisconnect
    from fastapi.websockets import WebSocketState
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    WebSocket = None
    WebSocketDisconnect = Exception

from .logger import get_logger
from .exceptions import CHSBDSException

logger = get_logger(__name__)


class WebSocketError(CHSBDSException):
    """WebSocket异常"""
    pass


class MessageType(str, Enum):
    """消息类型"""
    DATA_UPDATE = "data_update"
    ALERT = "alert"
    STATUS_UPDATE = "status_update"
    PING = "ping"
    PONG = "pong"
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    ERROR = "error"


class WebSocketConnectionManager:
    """WebSocket连接管理器"""

    def __init__(self):
        # 活动连接
        self.active_connections: Set[WebSocket] = set()

        # 订阅管理 {topic: set(websockets)}
        self.subscriptions: Dict[str, Set[WebSocket]] = {}

        # 连接元数据 {websocket: {metadata}}
        self.connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}

    async def connect(self, websocket: WebSocket, client_id: Optional[str] = None):
        """接受新连接"""
        await websocket.accept()
        self.active_connections.add(websocket)

        metadata = {
            'client_id': client_id or f"client_{id(websocket)}",
            'connected_at': datetime.now(),
            'subscriptions': set()
        }
        self.connection_metadata[websocket] = metadata

        logger.info(f"WebSocket connected: {metadata['client_id']}")

        # 发送欢迎消息
        await self.send_message(websocket, {
            'type': 'connection',
            'message': 'Connected to CHS-BDS WebSocket server',
            'client_id': metadata['client_id'],
            'timestamp': datetime.now().isoformat()
        })

    def disconnect(self, websocket: WebSocket):
        """断开连接"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

            # 从所有订阅中移除
            client_id = self.connection_metadata.get(websocket, {}).get('client_id', 'unknown')
            for topic in list(self.subscriptions.keys()):
                if websocket in self.subscriptions[topic]:
                    self.subscriptions[topic].remove(websocket)
                    if not self.subscriptions[topic]:
                        del self.subscriptions[topic]

            # 移除元数据
            if websocket in self.connection_metadata:
                del self.connection_metadata[websocket]

            logger.info(f"WebSocket disconnected: {client_id}")

    async def send_message(self, websocket: WebSocket, message: Dict[str, Any]):
        """发送消息到指定连接"""
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Failed to send message: {e}")

    async def broadcast(self, message: Dict[str, Any], exclude: Optional[Set[WebSocket]] = None):
        """广播消息到所有连接"""
        exclude = exclude or set()

        disconnected = []
        for connection in self.active_connections:
            if connection not in exclude:
                try:
                    await self.send_message(connection, message)
                except:
                    disconnected.append(connection)

        # 清理断开的连接
        for conn in disconnected:
            self.disconnect(conn)

    async def publish_to_topic(self, topic: str, message: Dict[str, Any]):
        """发布消息到特定主题"""
        if topic not in self.subscriptions:
            return

        message['topic'] = topic
        message['timestamp'] = datetime.now().isoformat()

        disconnected = []
        for websocket in self.subscriptions[topic]:
            try:
                await self.send_message(websocket, message)
            except:
                disconnected.append(websocket)

        # 清理断开的连接
        for conn in disconnected:
            self.disconnect(conn)

    def subscribe(self, websocket: WebSocket, topic: str):
        """订阅主题"""
        if topic not in self.subscriptions:
            self.subscriptions[topic] = set()

        self.subscriptions[topic].add(websocket)

        if websocket in self.connection_metadata:
            self.connection_metadata[websocket]['subscriptions'].add(topic)

        logger.info(f"Client subscribed to topic: {topic}")

    def unsubscribe(self, websocket: WebSocket, topic: str):
        """取消订阅"""
        if topic in self.subscriptions and websocket in self.subscriptions[topic]:
            self.subscriptions[topic].remove(websocket)

            if websocket in self.connection_metadata:
                self.connection_metadata[websocket]['subscriptions'].discard(topic)

            if not self.subscriptions[topic]:
                del self.subscriptions[topic]

            logger.info(f"Client unsubscribed from topic: {topic}")

    def get_stats(self) -> Dict[str, Any]:
        """获取连接统计信息"""
        return {
            'total_connections': len(self.active_connections),
            'total_subscriptions': sum(len(subs) for subs in self.subscriptions.values()),
            'topics': list(self.subscriptions.keys()),
            'topic_subscriber_count': {
                topic: len(subs) for topic, subs in self.subscriptions.items()
            }
        }


class WebSocketServer:
    """WebSocket服务器"""

    def __init__(self):
        if not FASTAPI_AVAILABLE:
            raise WebSocketError("FastAPI is required for WebSocket support")

        self.manager = WebSocketConnectionManager()
        self._running = False

    async def handle_client(self, websocket: WebSocket, client_id: Optional[str] = None):
        """处理客户端连接"""
        await self.manager.connect(websocket, client_id)

        try:
            while True:
                # 接收消息
                data = await websocket.receive_json()

                # 处理消息
                await self._process_message(websocket, data)

        except WebSocketDisconnect:
            self.manager.disconnect(websocket)
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            self.manager.disconnect(websocket)

    async def _process_message(self, websocket: WebSocket, data: Dict[str, Any]):
        """处理接收到的消息"""
        msg_type = data.get('type', '')

        if msg_type == MessageType.PING:
            # 响应ping
            await self.manager.send_message(websocket, {
                'type': MessageType.PONG,
                'timestamp': datetime.now().isoformat()
            })

        elif msg_type == MessageType.SUBSCRIBE:
            # 订阅主题
            topic = data.get('topic', '')
            if topic:
                self.manager.subscribe(websocket, topic)
                await self.manager.send_message(websocket, {
                    'type': 'subscription_confirmed',
                    'topic': topic,
                    'message': f'Subscribed to {topic}'
                })

        elif msg_type == MessageType.UNSUBSCRIBE:
            # 取消订阅
            topic = data.get('topic', '')
            if topic:
                self.manager.unsubscribe(websocket, topic)
                await self.manager.send_message(websocket, {
                    'type': 'unsubscription_confirmed',
                    'topic': topic,
                    'message': f'Unsubscribed from {topic}'
                })

        else:
            logger.warning(f"Unknown message type: {msg_type}")

    async def broadcast_data_update(self, module: str, data: Dict[str, Any]):
        """广播数据更新"""
        message = {
            'type': MessageType.DATA_UPDATE,
            'module': module,
            'data': data,
            'timestamp': datetime.now().isoformat()
        }

        await self.manager.publish_to_topic(f"data.{module}", message)
        await self.manager.publish_to_topic("data.all", message)

    async def broadcast_alert(self, alert: Dict[str, Any]):
        """广播告警"""
        message = {
            'type': MessageType.ALERT,
            'alert': alert,
            'timestamp': datetime.now().isoformat()
        }

        level = alert.get('level', '').lower()
        await self.manager.publish_to_topic(f"alerts.{level}", message)
        await self.manager.publish_to_topic("alerts.all", message)

    async def broadcast_status_update(self, status: Dict[str, Any]):
        """广播状态更新"""
        message = {
            'type': MessageType.STATUS_UPDATE,
            'status': status,
            'timestamp': datetime.now().isoformat()
        }

        await self.manager.publish_to_topic("system.status", message)

    def get_stats(self) -> Dict[str, Any]:
        """获取服务器统计信息"""
        return self.manager.get_stats()


# 全局WebSocket服务器实例
_websocket_server = None


def get_websocket_server() -> WebSocketServer:
    """获取WebSocket服务器单例"""
    global _websocket_server
    if _websocket_server is None:
        _websocket_server = WebSocketServer()
    return _websocket_server


# 主题常量
class Topics:
    """预定义主题"""
    DATA_ALL = "data.all"
    DATA_GNSS_IR = "data.gnss_ir"
    DATA_DEFORMATION = "data.deformation"
    DATA_PWV = "data.pwv"
    DATA_RAINFALL = "data.rainfall"

    ALERTS_ALL = "alerts.all"
    ALERTS_INFO = "alerts.info"
    ALERTS_WARNING = "alerts.warning"
    ALERTS_CRITICAL = "alerts.critical"

    SYSTEM_STATUS = "system.status"


if __name__ == "__main__":
    print("WebSocket Server module")
    print("\nAvailable topics:")
    for attr in dir(Topics):
        if not attr.startswith('_'):
            print(f"  - {getattr(Topics, attr)}")
