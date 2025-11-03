"""
System health monitoring implementation for Project Argus.

This module implements comprehensive system health monitoring including:
- Continuous component status checking
- Network connectivity monitoring and offline mode handling
- Predictive maintenance alerts and system diagnostics
"""

import asyncio
import logging
import threading
import time
import psutil
import socket
import subprocess
import json
from typing import Dict, Any, List, Optional, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque, defaultdict
import statistics

from shared.interfaces.health import IHealthMonitor
from shared.models.health import HealthStatus, ComponentHealth, ComponentType, ComponentStatus


@dataclass
class HealthCheckResult:
    """Result of a health check operation."""
    component_id: str
    success: bool
    metrics: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PredictiveAlert:
    """Predictive maintenance alert."""
    component_id: str
    alert_type: str
    predicted_failure_time: datetime
    confidence: float
    description: str
    recommended_actions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class HealthMonitor(IHealthMonitor):
    """
    Comprehensive system health monitoring implementation.
    
    Provides continuous monitoring of system components, network connectivity,
    predictive maintenance alerts, and offline mode handling.
    """
    
    def __init__(self, check_interval: int = 30, history_retention_hours: int = 24):
        """
        Initialize health monitor.
        
        Args:
            check_interval: Default interval between health checks in seconds
            history_retention_hours: How long to retain health history
        """
        self.check_interval = check_interval
        self.history_retention_hours = history_retention_hours
        self.logger = logging.getLogger("HealthMonitor")
        
        # Component registry
        self.components: Dict[str, ComponentHealth] = {}
        self.component_intervals: Dict[str, int] = {}
        self.health_checks: Dict[str, Callable[[], Dict[str, Any]]] = {}
        self.alert_thresholds: Dict[ComponentType, Dict[str, float]] = {}
        
        # Health history for trend analysis
        self.health_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # System state
        self.system_health = HealthStatus()
        self.is_running = False
        self.is_offline_mode = False
        self.last_network_check = datetime.now()
        self.network_check_interval = 60  # seconds
        
        # Threading
        self.monitor_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        
        # Predictive analytics
        self.failure_patterns: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.trend_windows = {
            'short': timedelta(hours=1),
            'medium': timedelta(hours=6),
            'long': timedelta(hours=24)
        }
        
        # Default thresholds
        self._setup_default_thresholds()
        
        # System metrics tracking
        self.system_start_time = datetime.now()
        self.last_metrics_update = datetime.now()
        
    def _setup_default_thresholds(self) -> None:
        """Setup default alert thresholds for different component types."""
        self.alert_thresholds = {
            ComponentType.CAMERA: {
                'cpu_usage': 80.0,
                'memory_usage': 85.0,
                'temperature': 70.0,
                'error_rate': 5.0
            },
            ComponentType.EDGE_NODE: {
                'cpu_usage': 85.0,
                'memory_usage': 90.0,
                'disk_usage': 85.0,
                'temperature': 75.0,
                'network_latency': 1000.0
            },
            ComponentType.API_SERVICE: {
                'cpu_usage': 80.0,
                'memory_usage': 85.0,
                'response_time': 5000.0,
                'error_rate': 1.0
            },
            ComponentType.DATABASE: {
                'cpu_usage': 75.0,
                'memory_usage': 80.0,
                'disk_usage': 80.0,
                'connection_count': 100.0
            }
        }
    
    def register_component(self, component_id: str, component_type: ComponentType, 
                          check_interval_seconds: int = 30) -> None:
        """Register a component for health monitoring."""
        with self.lock:
            component = ComponentHealth(
                component_id=component_id,
                component_type=component_type,
                status=ComponentStatus.HEALTHY
            )
            
            self.components[component_id] = component
            self.component_intervals[component_id] = check_interval_seconds
            self.system_health.add_component(component)
            
            self.logger.info(f"Registered component {component_id} of type {component_type.value}")
    
    def unregister_component(self, component_id: str) -> None:
        """Unregister a component from health monitoring."""
        with self.lock:
            if component_id in self.components:
                del self.components[component_id]
                del self.component_intervals[component_id]
                if component_id in self.health_checks:
                    del self.health_checks[component_id]
                
                self.system_health.remove_component(component_id)
                
                self.logger.info(f"Unregistered component {component_id}")
    
    def update_component_health(self, component_id: str, health_data: Dict[str, Any]) -> None:
        """Update health status for a component."""
        with self.lock:
            if component_id not in self.components:
                self.logger.warning(f"Component {component_id} not registered")
                return
            
            component = self.components[component_id]
            
            # Update metrics
            component.cpu_usage = health_data.get('cpu_usage')
            component.memory_usage = health_data.get('memory_usage')
            component.disk_usage = health_data.get('disk_usage')
            component.network_latency = health_data.get('network_latency')
            component.temperature = health_data.get('temperature')
            component.uptime_seconds = health_data.get('uptime_seconds')
            
            # Handle error messages from health check
            if 'error_messages' in health_data:
                for error_msg in health_data['error_messages']:
                    component.add_error(error_msg)
            
            # Handle warning messages from health check
            if 'warning_messages' in health_data:
                for warning_msg in health_data['warning_messages']:
                    component.add_warning(warning_msg)
            
            # Update metadata - include all health_data except standard metrics
            standard_metrics = {'cpu_usage', 'memory_usage', 'disk_usage', 'network_latency', 
                               'temperature', 'uptime_seconds', 'metadata', 'error_messages', 'warning_messages'}
            for key, value in health_data.items():
                if key not in standard_metrics:
                    component.metadata[key] = value
            
            # Also update explicit metadata if provided
            component.metadata.update(health_data.get('metadata', {}))
            
            # Store current error/warning messages before threshold evaluation
            existing_errors = component.error_messages.copy()
            existing_warnings = component.warning_messages.copy()
            
            # Check thresholds and update status
            self._evaluate_component_health(component)
            
            # Restore and merge error/warning messages
            component.error_messages.extend(existing_errors)
            component.warning_messages.extend(existing_warnings)
            
            # Update system health
            self.system_health.add_component(component)
            
            # Store in history
            self._store_health_history(component_id, health_data)
    
    def get_component_health(self, component_id: str) -> Optional[ComponentHealth]:
        """Get health status of specific component."""
        return self.components.get(component_id)
    
    def get_system_health(self) -> HealthStatus:
        """Get overall system health status."""
        with self.lock:
            # Update system-wide metrics
            self._update_system_metrics()
            return self.system_health
    
    def get_unhealthy_components(self) -> List[ComponentHealth]:
        """Get all components that are not healthy."""
        return [comp for comp in self.components.values() if not comp.is_healthy()]
    
    def register_health_check(self, component_id: str, 
                             check_function: Callable[[], Dict[str, Any]]) -> None:
        """Register custom health check function for component."""
        self.health_checks[component_id] = check_function
        self.logger.info(f"Registered health check for component {component_id}")
    
    def run_health_checks(self) -> None:
        """Run all registered health checks."""
        for component_id, check_function in self.health_checks.items():
            try:
                health_data = check_function()
                self.update_component_health(component_id, health_data)
            except Exception as e:
                self.logger.error(f"Health check failed for {component_id}: {e}")
                
                # Mark component as having issues
                if component_id in self.components:
                    component = self.components[component_id]
                    component.add_error(f"Health check failed: {str(e)}")
                    self.system_health.add_component(component)
    
    def set_alert_thresholds(self, component_type: ComponentType, 
                           thresholds: Dict[str, float]) -> None:
        """Set alert thresholds for component type."""
        self.alert_thresholds[component_type] = thresholds
        self.logger.info(f"Updated thresholds for {component_type.value}: {thresholds}")
    
    def get_health_history(self, component_id: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get health history for a component."""
        if component_id not in self.health_history:
            return []
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        history = self.health_history[component_id]
        
        return [entry for entry in history if entry['timestamp'] > cutoff_time]
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get aggregated system performance metrics."""
        with self.lock:
            healthy_components = sum(1 for comp in self.components.values() if comp.is_healthy())
            total_components = len(self.components)
            
            # Calculate averages
            cpu_values = [comp.cpu_usage for comp in self.components.values() if comp.cpu_usage is not None]
            memory_values = [comp.memory_usage for comp in self.components.values() if comp.memory_usage is not None]
            disk_values = [comp.disk_usage for comp in self.components.values() if comp.disk_usage is not None]
            
            return {
                'system_uptime_hours': (datetime.now() - self.system_start_time).total_seconds() / 3600,
                'total_components': total_components,
                'healthy_components': healthy_components,
                'system_availability': (healthy_components / total_components * 100) if total_components > 0 else 0,
                'avg_cpu_usage': statistics.mean(cpu_values) if cpu_values else None,
                'avg_memory_usage': statistics.mean(memory_values) if memory_values else None,
                'avg_disk_usage': statistics.mean(disk_values) if disk_values else None,
                'offline_mode': self.is_offline_mode,
                'network_connectivity': self._check_network_connectivity(),
                'last_update': datetime.now().isoformat()
            }
    
    def predict_failures(self, component_id: str) -> Dict[str, Any]:
        """Predict potential component failures based on trends."""
        if component_id not in self.health_history:
            return {'prediction': 'insufficient_data'}
        
        try:
            history = list(self.health_history[component_id])
            if len(history) < 10:
                return {'prediction': 'insufficient_data'}
            
            # Analyze trends for different metrics
            predictions = {}
            
            # CPU usage trend
            cpu_trend = self._analyze_metric_trend(history, 'cpu_usage')
            if cpu_trend['trend'] == 'increasing' and cpu_trend['rate'] > 1.0:
                predictions['cpu_overload'] = {
                    'risk': 'high' if cpu_trend['rate'] > 5.0 else 'medium',
                    'estimated_time_to_failure': self._estimate_failure_time(cpu_trend, 90.0),
                    'confidence': cpu_trend['confidence']
                }
            
            # Memory usage trend
            memory_trend = self._analyze_metric_trend(history, 'memory_usage')
            if memory_trend['trend'] == 'increasing' and memory_trend['rate'] > 1.0:
                predictions['memory_exhaustion'] = {
                    'risk': 'high' if memory_trend['rate'] > 3.0 else 'medium',
                    'estimated_time_to_failure': self._estimate_failure_time(memory_trend, 95.0),
                    'confidence': memory_trend['confidence']
                }
            
            # Temperature trend
            temp_trend = self._analyze_metric_trend(history, 'temperature')
            if temp_trend['trend'] == 'increasing' and temp_trend['rate'] > 0.5:
                predictions['overheating'] = {
                    'risk': 'high' if temp_trend['rate'] > 2.0 else 'medium',
                    'estimated_time_to_failure': self._estimate_failure_time(temp_trend, 85.0),
                    'confidence': temp_trend['confidence']
                }
            
            # Error rate trend
            error_trend = self._analyze_error_trend(history)
            if error_trend['trend'] == 'increasing':
                predictions['system_instability'] = {
                    'risk': 'high' if error_trend['rate'] > 2.0 else 'medium',
                    'confidence': error_trend['confidence']
                }
            
            return {
                'component_id': component_id,
                'predictions': predictions,
                'analysis_timestamp': datetime.now().isoformat(),
                'data_points_analyzed': len(history)
            }
            
        except Exception as e:
            self.logger.error(f"Error in failure prediction for {component_id}: {e}")
            return {'prediction': 'analysis_error', 'error': str(e)}
    
    def start_monitoring(self) -> None:
        """Start the health monitoring system."""
        if self.is_running:
            self.logger.warning("Health monitoring is already running")
            return
        
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("Health monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop the health monitoring system."""
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        self.logger.info("Health monitoring stopped")
    
    def enable_offline_mode(self) -> None:
        """Enable offline mode for network disconnection scenarios."""
        self.is_offline_mode = True
        self.logger.info("Offline mode enabled")
    
    def disable_offline_mode(self) -> None:
        """Disable offline mode when network connectivity is restored."""
        self.is_offline_mode = False
        self.logger.info("Offline mode disabled")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop running in separate thread."""
        while self.is_running:
            try:
                # Run health checks
                self.run_health_checks()
                
                # Check network connectivity
                if datetime.now() - self.last_network_check > timedelta(seconds=self.network_check_interval):
                    self._check_and_handle_network_connectivity()
                    self.last_network_check = datetime.now()
                
                # Run system-level health checks
                self._run_system_health_checks()
                
                # Clean up old history
                self._cleanup_old_history()
                
                # Generate predictive alerts
                self._generate_predictive_alerts()
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.check_interval)
    
    def _run_system_health_checks(self) -> None:
        """Run system-level health checks."""
        try:
            # System resource checks
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network checks
            network_stats = psutil.net_io_counters()
            
            # Process checks
            process_count = len(psutil.pids())
            
            # Update system health
            system_data = {
                'cpu_usage': cpu_percent,
                'memory_usage': memory.percent,
                'disk_usage': disk.percent,
                'process_count': process_count,
                'network_bytes_sent': network_stats.bytes_sent,
                'network_bytes_recv': network_stats.bytes_recv,
                'timestamp': datetime.now()
            }
            
            # Check for system-level issues
            if cpu_percent > 90:
                self.logger.warning(f"High system CPU usage: {cpu_percent}%")
            
            if memory.percent > 90:
                self.logger.warning(f"High system memory usage: {memory.percent}%")
            
            if disk.percent > 90:
                self.logger.warning(f"High system disk usage: {disk.percent}%")
            
            # Store system metrics
            self._store_health_history('system', system_data)
            
        except Exception as e:
            self.logger.error(f"Error in system health checks: {e}")
    
    def _check_and_handle_network_connectivity(self) -> None:
        """Check network connectivity and handle offline mode."""
        is_connected = self._check_network_connectivity()
        
        if not is_connected and not self.is_offline_mode:
            self.enable_offline_mode()
            self.logger.warning("Network connectivity lost - enabling offline mode")
        elif is_connected and self.is_offline_mode:
            self.disable_offline_mode()
            self.logger.info("Network connectivity restored - disabling offline mode")
    
    def _check_network_connectivity(self) -> bool:
        """Check if network connectivity is available."""
        try:
            # Try to connect to Google DNS
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True
        except OSError:
            try:
                # Try alternative DNS
                socket.create_connection(("1.1.1.1", 53), timeout=3)
                return True
            except OSError:
                return False
    
    def _evaluate_component_health(self, component: ComponentHealth) -> None:
        """Evaluate component health against thresholds."""
        component_type = component.component_type
        thresholds = self.alert_thresholds.get(component_type, {})
        
        # Clear previous status
        component.status = ComponentStatus.HEALTHY
        component.warning_messages.clear()
        component.error_messages.clear()
        
        # Check each metric against thresholds
        if component.cpu_usage is not None:
            cpu_threshold = thresholds.get('cpu_usage', 85.0)
            if component.cpu_usage > cpu_threshold:
                if component.cpu_usage > cpu_threshold * 1.1:
                    component.add_error(f"Critical CPU usage: {component.cpu_usage}%")
                else:
                    component.add_warning(f"High CPU usage: {component.cpu_usage}%")
        
        if component.memory_usage is not None:
            memory_threshold = thresholds.get('memory_usage', 90.0)
            if component.memory_usage > memory_threshold:
                if component.memory_usage > memory_threshold * 1.05:
                    component.add_error(f"Critical memory usage: {component.memory_usage}%")
                else:
                    component.add_warning(f"High memory usage: {component.memory_usage}%")
        
        if component.temperature is not None:
            temp_threshold = thresholds.get('temperature', 75.0)
            if component.temperature > temp_threshold:
                if component.temperature > temp_threshold * 1.1:
                    component.add_error(f"Critical temperature: {component.temperature}°C")
                else:
                    component.add_warning(f"High temperature: {component.temperature}°C")
        
        component.last_check = datetime.now()
    
    def _store_health_history(self, component_id: str, health_data: Dict[str, Any]) -> None:
        """Store health data in history for trend analysis."""
        entry = {
            'timestamp': datetime.now(),
            'data': health_data.copy()
        }
        
        self.health_history[component_id].append(entry)
    
    def _cleanup_old_history(self) -> None:
        """Clean up old health history data."""
        cutoff_time = datetime.now() - timedelta(hours=self.history_retention_hours)
        
        for component_id in self.health_history:
            history = self.health_history[component_id]
            # Remove old entries
            while history and history[0]['timestamp'] < cutoff_time:
                history.popleft()
    
    def _update_system_metrics(self) -> None:
        """Update system-wide metrics in HealthStatus."""
        # Count cameras and edge nodes
        cameras = [comp for comp in self.components.values() 
                  if comp.component_type == ComponentType.CAMERA]
        edge_nodes = [comp for comp in self.components.values() 
                     if comp.component_type == ComponentType.EDGE_NODE]
        
        self.system_health.total_cameras = len(cameras)
        self.system_health.active_cameras = sum(1 for cam in cameras if cam.is_healthy())
        self.system_health.total_edge_nodes = len(edge_nodes)
        self.system_health.active_edge_nodes = sum(1 for node in edge_nodes if node.is_healthy())
        
        # Update network connectivity
        self.system_health.network_connectivity = not self.is_offline_mode
        self.system_health.internet_connectivity = self._check_network_connectivity()
        
        # Update timestamp
        self.system_health.timestamp = datetime.now()
    
    def _analyze_metric_trend(self, history: List[Dict], metric_name: str) -> Dict[str, Any]:
        """Analyze trend for a specific metric."""
        values = []
        timestamps = []
        
        for entry in history[-50:]:  # Last 50 data points
            if metric_name in entry['data'] and entry['data'][metric_name] is not None:
                values.append(entry['data'][metric_name])
                timestamps.append(entry['timestamp'])
        
        if len(values) < 5:
            return {'trend': 'unknown', 'confidence': 0.0}
        
        # Simple linear regression for trend
        n = len(values)
        x = list(range(n))
        
        # Calculate slope
        x_mean = sum(x) / n
        y_mean = sum(values) / n
        
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return {'trend': 'stable', 'confidence': 0.5}
        
        slope = numerator / denominator
        
        # Determine trend
        if abs(slope) < 0.1:
            trend = 'stable'
        elif slope > 0:
            trend = 'increasing'
        else:
            trend = 'decreasing'
        
        # Calculate confidence based on R-squared
        y_pred = [y_mean + slope * (i - x_mean) for i in x]
        ss_res = sum((values[i] - y_pred[i]) ** 2 for i in range(n))
        ss_tot = sum((values[i] - y_mean) ** 2 for i in range(n))
        
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        confidence = max(0, min(1, r_squared))
        
        return {
            'trend': trend,
            'rate': abs(slope),
            'confidence': confidence,
            'data_points': n
        }
    
    def _analyze_error_trend(self, history: List[Dict]) -> Dict[str, Any]:
        """Analyze error rate trend."""
        error_counts = []
        
        for entry in history[-20:]:  # Last 20 data points
            error_count = entry['data'].get('error_count', 0)
            error_counts.append(error_count)
        
        if len(error_counts) < 5:
            return {'trend': 'unknown', 'confidence': 0.0}
        
        # Calculate rate of change
        recent_avg = sum(error_counts[-5:]) / 5
        older_avg = sum(error_counts[:5]) / 5
        
        if older_avg == 0:
            rate = recent_avg
        else:
            rate = (recent_avg - older_avg) / older_avg
        
        trend = 'increasing' if rate > 0.2 else 'stable' if rate > -0.2 else 'decreasing'
        confidence = min(1.0, abs(rate))
        
        return {
            'trend': trend,
            'rate': rate,
            'confidence': confidence
        }
    
    def _estimate_failure_time(self, trend_data: Dict, failure_threshold: float) -> Optional[str]:
        """Estimate time to failure based on trend."""
        if trend_data['trend'] != 'increasing' or trend_data['rate'] <= 0:
            return None
        
        # Simple linear extrapolation
        current_value = 50.0  # Assume current value (would be actual in real implementation)
        rate_per_hour = trend_data['rate']  # Assume rate is per hour
        
        if rate_per_hour <= 0:
            return None
        
        hours_to_failure = (failure_threshold - current_value) / rate_per_hour
        
        if hours_to_failure <= 0:
            return "immediate"
        elif hours_to_failure < 24:
            return f"{int(hours_to_failure)} hours"
        else:
            return f"{int(hours_to_failure / 24)} days"
    
    def _generate_predictive_alerts(self) -> None:
        """Generate predictive maintenance alerts."""
        for component_id in self.components:
            try:
                predictions = self.predict_failures(component_id)
                
                if 'predictions' in predictions:
                    for failure_type, prediction in predictions['predictions'].items():
                        if prediction['risk'] == 'high':
                            self.logger.warning(
                                f"Predictive alert for {component_id}: "
                                f"{failure_type} risk is high"
                            )
                            
            except Exception as e:
                self.logger.error(f"Error generating predictive alerts for {component_id}: {e}")