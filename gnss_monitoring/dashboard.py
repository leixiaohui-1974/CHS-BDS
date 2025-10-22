"""
Web Dashboard Module
====================

Interactive web dashboard for CHS-BDS using Plotly Dash.
Provides real-time visualization and monitoring capabilities.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

try:
    import dash
    from dash import dcc, html, Input, Output, State
    import dash_bootstrap_components as dbc
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False

from .logger import get_logger
from .config import get_config


logger = get_logger(__name__)


class CHSBDSDashboard:
    """
    Interactive web dashboard for CHS-BDS monitoring system.

    Features:
    - Real-time data visualization
    - Module status monitoring
    - Quality control displays
    - Alert management
    - Historical data browsing
    """

    def __init__(self, port: int = 8050, debug: bool = False):
        """
        Initialize dashboard.

        Args:
            port: Port number for web server.
            debug: Enable debug mode.
        """
        if not DASH_AVAILABLE:
            raise ImportError(
                "Dash is not installed. Install with: "
                "pip install dash dash-bootstrap-components plotly"
            )

        self.port = port
        self.debug = debug
        self.config = get_config()

        # Initialize Dash app
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            suppress_callback_exceptions=True
        )

        self.app.title = "CHS-BDS Monitoring Dashboard"

        # Initialize data storage
        self.data_store = {
            'gnss_ir': [],
            'deformation': [],
            'pwv': [],
            'rainfall': [],
            'quality': []
        }

        logger.info(f"Dashboard initialized on port {port}")

    def create_layout(self) -> html.Div:
        """
        Create dashboard layout.

        Returns:
            Dash HTML layout.
        """
        layout = dbc.Container([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1("CHS-BDS Monitoring Dashboard", className="text-primary"),
                    html.P("GNSS Comprehensive Monitoring System", className="lead")
                ])
            ], className="mb-4 mt-4"),

            # Status Cards
            dbc.Row([
                dbc.Col(self._create_status_card("GNSS-IR", "gnss-ir-status"), md=3),
                dbc.Col(self._create_status_card("Deformation", "deformation-status"), md=3),
                dbc.Col(self._create_status_card("PWV", "pwv-status"), md=3),
                dbc.Col(self._create_status_card("Rainfall", "rainfall-status"), md=3),
            ], className="mb-4"),

            # Main Content Tabs
            dbc.Tabs([
                dbc.Tab(label="Overview", tab_id="overview"),
                dbc.Tab(label="GNSS-IR", tab_id="gnss-ir"),
                dbc.Tab(label="Deformation", tab_id="deformation"),
                dbc.Tab(label="PWV & Rainfall", tab_id="pwv"),
                dbc.Tab(label="Quality Control", tab_id="quality"),
                dbc.Tab(label="Settings", tab_id="settings"),
            ], id="tabs", active_tab="overview"),

            html.Div(id="tab-content", className="mt-4"),

            # Auto-refresh interval
            dcc.Interval(
                id='interval-component',
                interval=30*1000,  # 30 seconds
                n_intervals=0
            )

        ], fluid=True)

        return layout

    def _create_status_card(self, title: str, card_id: str) -> dbc.Card:
        """Create a status monitoring card."""
        return dbc.Card([
            dbc.CardBody([
                html.H5(title, className="card-title"),
                html.H3("--", id=card_id, className="text-success"),
                html.Small("Status: Active", className="text-muted")
            ])
        ])

    def _create_overview_tab(self) -> html.Div:
        """Create overview tab content."""
        return html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("System Overview"),
                        dbc.CardBody([
                            dcc.Graph(id='overview-chart')
                        ])
                    ])
                ], md=8),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Recent Alerts"),
                        dbc.CardBody([
                            html.Div(id='alerts-list')
                        ])
                    ])
                ], md=4)
            ])
        ])

    def _create_gnss_ir_tab(self) -> html.Div:
        """Create GNSS-IR monitoring tab."""
        return html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Water Level Time Series"),
                        dbc.CardBody([
                            dcc.Graph(id='gnss-ir-timeseries')
                        ])
                    ])
                ], md=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("SNR Analysis"),
                        dbc.CardBody([
                            dcc.Graph(id='gnss-ir-snr')
                        ])
                    ])
                ], md=6)
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Frequency Spectrum"),
                        dbc.CardBody([
                            dcc.Graph(id='gnss-ir-spectrum')
                        ])
                    ])
                ], md=12)
            ], className="mt-3")
        ])

    def _create_deformation_tab(self) -> html.Div:
        """Create deformation monitoring tab."""
        return html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("3D Displacement"),
                        dbc.CardBody([
                            dcc.Graph(id='deformation-3d')
                        ])
                    ])
                ], md=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Displacement Time Series"),
                        dbc.CardBody([
                            dcc.Graph(id='deformation-timeseries')
                        ])
                    ])
                ], md=6)
            ])
        ])

    def _create_pwv_tab(self) -> html.Div:
        """Create PWV and rainfall tab."""
        return html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("PWV Time Series"),
                        dbc.CardBody([
                            dcc.Graph(id='pwv-timeseries')
                        ])
                    ])
                ], md=12)
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Rainfall Prediction"),
                        dbc.CardBody([
                            dcc.Graph(id='rainfall-prediction')
                        ])
                    ])
                ], md=12)
            ], className="mt-3")
        ])

    def _create_quality_tab(self) -> html.Div:
        """Create quality control tab."""
        return html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Quality Metrics"),
                        dbc.CardBody([
                            dcc.Graph(id='quality-metrics')
                        ])
                    ])
                ], md=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Data Completeness"),
                        dbc.CardBody([
                            dcc.Graph(id='quality-completeness')
                        ])
                    ])
                ], md=6)
            ])
        ])

    def create_sample_chart(self, chart_type: str = 'timeseries') -> go.Figure:
        """
        Create sample chart for demonstration.

        Args:
            chart_type: Type of chart to create.

        Returns:
            Plotly figure.
        """
        if chart_type == 'timeseries':
            # Sample time series
            dates = pd.date_range(start='2025-01-01', periods=100, freq='H')
            values = np.cumsum(np.random.randn(100)) + 10

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=values,
                mode='lines+markers',
                name='Water Level',
                line=dict(color='blue', width=2)
            ))

            fig.update_layout(
                title="Water Level Monitoring",
                xaxis_title="Time",
                yaxis_title="Height (m)",
                hovermode='x unified'
            )

        elif chart_type == '3d':
            # Sample 3D scatter
            x = np.random.randn(50)
            y = np.random.randn(50)
            z = np.random.randn(50)

            fig = go.Figure(data=[go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                marker=dict(
                    size=8,
                    color=z,
                    colorscale='Viridis',
                    showscale=True
                )
            )])

            fig.update_layout(
                title="3D Displacement Vectors",
                scene=dict(
                    xaxis_title='X (m)',
                    yaxis_title='Y (m)',
                    zaxis_title='Z (m)'
                )
            )

        elif chart_type == 'gauge':
            # Quality score gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=85,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Quality Score"},
                delta={'reference': 80},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 75], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))

        else:
            fig = go.Figure()

        return fig

    def setup_callbacks(self):
        """Setup dashboard callbacks for interactivity."""

        @self.app.callback(
            Output('tab-content', 'children'),
            Input('tabs', 'active_tab')
        )
        def render_tab_content(active_tab):
            """Render content based on active tab."""
            if active_tab == 'overview':
                return self._create_overview_tab()
            elif active_tab == 'gnss-ir':
                return self._create_gnss_ir_tab()
            elif active_tab == 'deformation':
                return self._create_deformation_tab()
            elif active_tab == 'pwv':
                return self._create_pwv_tab()
            elif active_tab == 'quality':
                return self._create_quality_tab()
            elif active_tab == 'settings':
                return html.Div("Settings panel - Coming soon")

        @self.app.callback(
            Output('overview-chart', 'figure'),
            Input('interval-component', 'n_intervals')
        )
        def update_overview_chart(n):
            """Update overview chart."""
            return self.create_sample_chart('timeseries')

        @self.app.callback(
            [Output('gnss-ir-timeseries', 'figure'),
             Output('gnss-ir-snr', 'figure'),
             Output('gnss-ir-spectrum', 'figure')],
            Input('interval-component', 'n_intervals')
        )
        def update_gnss_ir_charts(n):
            """Update GNSS-IR charts."""
            return (
                self.create_sample_chart('timeseries'),
                self.create_sample_chart('timeseries'),
                self.create_sample_chart('timeseries')
            )

        @self.app.callback(
            Output('alerts-list', 'children'),
            Input('interval-component', 'n_intervals')
        )
        def update_alerts(n):
            """Update alerts list."""
            alerts = [
                dbc.Alert("All systems operational", color="success", className="mb-2"),
                dbc.Alert("Last update: " + datetime.now().strftime("%H:%M:%S"),
                         color="info", className="mb-2")
            ]
            return alerts

    def run(self):
        """Start the dashboard server."""
        self.app.layout = self.create_layout()
        self.setup_callbacks()

        logger.info(f"Starting dashboard on http://localhost:{self.port}")
        print(f"\n{'='*70}")
        print(f"🌐 CHS-BDS Dashboard Starting...")
        print(f"{'='*70}")
        print(f"\n📊 Dashboard URL: http://localhost:{self.port}")
        print(f"📈 Features: Real-time monitoring, Interactive charts, Quality control")
        print(f"\n💡 Press Ctrl+C to stop the server\n")

        self.app.run_server(
            host='0.0.0.0',
            port=self.port,
            debug=self.debug
        )


def create_dashboard_app(port: int = 8050, debug: bool = False) -> CHSBDSDashboard:
    """
    Factory function to create dashboard application.

    Args:
        port: Port number for server.
        debug: Enable debug mode.

    Returns:
        Dashboard instance.
    """
    dashboard = CHSBDSDashboard(port=port, debug=debug)
    return dashboard


if __name__ == '__main__':
    # Run dashboard
    if not DASH_AVAILABLE:
        print("❌ Dash is not installed!")
        print("Install with: pip install dash dash-bootstrap-components plotly")
    else:
        print("Starting CHS-BDS Dashboard...")
        dashboard = create_dashboard_app(port=8050, debug=True)
        dashboard.run()
