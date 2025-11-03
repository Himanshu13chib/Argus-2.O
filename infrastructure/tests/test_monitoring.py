#!/usr/bin/env python3
"""
Monitoring system tests for Project Argus
Tests Prometheus metrics, Grafana dashboards, and alerting functionality
"""

import pytest
import requests
import time
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional


class TestPrometheusMetrics:
    """Test Prometheus metrics collection and alerting rules"""
    
    @pytest.fixture(scope="class")
    def prometheus_url(self):
        """Prometheus base URL"""
        return "http://localhost:9090"
    
    def test_prometheus_config_valid(self):
        """Test Prometheus configuration is valid"""
        config_file = "infrastructure/monitoring/prometheus.yml"
        if Path(config_file).exists():
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                
                # Check required sections
                assert "global" in config
                assert "scrape_configs" in config
                assert "rule_files" in config
                assert "alerting" in config
                
                # Check scrape configs
                scrape_configs = config["scrape_configs"]
                assert len(scrape_configs) > 0
                
                # Check for Argus-specific jobs
                job_names = [job["job_name"] for job in scrape_configs]
                expected_jobs = [
                    "argus-api-gateway",
                    "argus-auth-service", 
                    "argus-alert-service",
                    "argus-tracking-service",
                    "argus-evidence-service"
                ]
                
                for job in expected_jobs:
                    assert job in job_names
    
    def test_alerting_rules_valid(self):
        """Test alerting rules are valid"""
        rules_file = "infrastructure/monitoring/alerting-rules.yml"
        if Path(rules_file).exists():
            with open(rules_file, 'r') as f:
                rules = yaml.safe_load(f)
                
                assert "groups" in rules
                assert len(rules["groups"]) > 0
                
                for group in rules["groups"]:
                    assert "name" in group
                    assert "rules" in group
                    
                    for rule in group["rules"]:
                        assert "alert" in rule
                        assert "expr" in rule
                        assert "labels" in rule
                        assert "annotations" in rule
                        
                        # Check required labels
                        labels = rule["labels"]
                        assert "severity" in labels
                        assert "component" in labels
    
    def test_prometheus_targets(self, prometheus_url):
        """Test Prometheus can reach its targets"""
        try:
            response = requests.get(f"{prometheus_url}/api/v1/targets", timeout=10)
            assert response.status_code == 200
            
            data = response.json()
            assert data["status"] == "success"
            
            # Check that we have active targets
            active_targets = data["data"]["activeTargets"]
            assert len(active_targets) > 0
            
            # Check for critical services
            target_jobs = [target["labels"]["job"] for target in active_targets]
            critical_jobs = ["prometheus", "node-exporter"]
            
            for job in critical_jobs:
                if job in target_jobs:
                    # Find the target and check its health
                    target = next(t for t in active_targets if t["labels"]["job"] == job)
                    assert target["health"] == "up"
                    
        except requests.exceptions.RequestException:
            pytest.skip("Prometheus not accessible for target testing")
    
    def test_custom_metrics_available(self, prometheus_url):
        """Test custom Argus metrics are available"""
        expected_metrics = [
            "argus_detections_total",
            "argus_alerts_total", 
            "argus_detection_duration_seconds",
            "argus_false_positive_alerts_total",
            "argus_virtual_line_crossings_total"
        ]
        
        try:
            for metric in expected_metrics:
                response = requests.get(
                    f"{prometheus_url}/api/v1/query",
                    params={"query": metric},
                    timeout=10
                )
                assert response.status_code == 200
                
                data = response.json()
                assert data["status"] == "success"
                # Note: data["data"]["result"] might be empty if no data points exist yet
                
        except requests.exceptions.RequestException:
            pytest.skip("Prometheus not accessible for metrics testing")


class TestGrafanaDashboards:
    """Test Grafana dashboard configuration and accessibility"""
    
    @pytest.fixture(scope="class")
    def grafana_url(self):
        """Grafana base URL"""
        return "http://localhost:3001"
    
    def test_grafana_datasources_config(self):
        """Test Grafana datasources configuration"""
        config_file = "infrastructure/monitoring/grafana-datasources.yml"
        if Path(config_file).exists():
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                
                assert "datasources" in config
                datasources = config["datasources"]
                
                # Check for required datasources
                datasource_names = [ds["name"] for ds in datasources]
                expected_datasources = ["Prometheus", "Loki", "Jaeger"]
                
                for ds in expected_datasources:
                    assert ds in datasource_names
                
                # Check Prometheus datasource
                prometheus_ds = next(ds for ds in datasources if ds["name"] == "Prometheus")
                assert prometheus_ds["type"] == "prometheus"
                assert prometheus_ds["url"] == "http://prometheus:9090"
    
    def test_dashboard_json_valid(self):
        """Test dashboard JSON files are valid"""
        dashboard_files = [
            "infrastructure/monitoring/grafana-dashboards/argus-overview.json",
            "infrastructure/monitoring/grafana-dashboards/detection-performance.json"
        ]
        
        for dashboard_file in dashboard_files:
            if Path(dashboard_file).exists():
                with open(dashboard_file, 'r') as f:
                    dashboard = json.load(f)
                    
                    # Check required dashboard structure
                    assert "dashboard" in dashboard
                    dash = dashboard["dashboard"]
                    
                    assert "title" in dash
                    assert "panels" in dash
                    assert len(dash["panels"]) > 0
                    
                    # Check panels have required fields
                    for panel in dash["panels"]:
                        assert "id" in panel
                        assert "title" in panel
                        assert "type" in panel
                        if "targets" in panel:
                            for target in panel["targets"]:
                                assert "expr" in target  # Prometheus query
    
    def test_grafana_health(self, grafana_url):
        """Test Grafana health endpoint"""
        try:
            response = requests.get(f"{grafana_url}/api/health", timeout=10)
            assert response.status_code == 200
            
            health_data = response.json()
            assert health_data["database"] == "ok"
            
        except requests.exceptions.RequestException:
            pytest.skip("Grafana not accessible for health testing")


class TestAlertManager:
    """Test AlertManager configuration and functionality"""
    
    @pytest.fixture(scope="class")
    def alertmanager_url(self):
        """AlertManager base URL"""
        return "http://localhost:9093"
    
    def test_alertmanager_config_valid(self):
        """Test AlertManager configuration is valid"""
        config_file = "infrastructure/monitoring/alertmanager.yml"
        if Path(config_file).exists():
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                
                # Check required sections
                assert "global" in config
                assert "route" in config
                assert "receivers" in config
                
                # Check route configuration
                route = config["route"]
                assert "group_by" in route
                assert "receiver" in route
                assert "routes" in route
                
                # Check receivers
                receivers = config["receivers"]
                assert len(receivers) > 0
                
                # Check for critical alert receiver
                receiver_names = [r["name"] for r in receivers]
                assert "critical-alerts" in receiver_names
    
    def test_alertmanager_status(self, alertmanager_url):
        """Test AlertManager status endpoint"""
        try:
            response = requests.get(f"{alertmanager_url}/api/v1/status", timeout=10)
            assert response.status_code == 200
            
            status_data = response.json()
            assert status_data["status"] == "success"
            
        except requests.exceptions.RequestException:
            pytest.skip("AlertManager not accessible for status testing")
    
    def test_alert_routing_config(self, alertmanager_url):
        """Test alert routing configuration"""
        try:
            response = requests.get(f"{alertmanager_url}/api/v1/alerts", timeout=10)
            assert response.status_code == 200
            
            # The response should be valid JSON even if no alerts are active
            alerts_data = response.json()
            assert "data" in alerts_data
            
        except requests.exceptions.RequestException:
            pytest.skip("AlertManager not accessible for routing testing")


class TestJaegerTracing:
    """Test Jaeger distributed tracing functionality"""
    
    @pytest.fixture(scope="class")
    def jaeger_url(self):
        """Jaeger base URL"""
        return "http://localhost:16686"
    
    def test_jaeger_health(self, jaeger_url):
        """Test Jaeger health endpoint"""
        try:
            response = requests.get(f"{jaeger_url}/", timeout=10)
            assert response.status_code == 200
            assert "Jaeger" in response.text
            
        except requests.exceptions.RequestException:
            pytest.skip("Jaeger not accessible for health testing")
    
    def test_jaeger_api_services(self, jaeger_url):
        """Test Jaeger API services endpoint"""
        try:
            response = requests.get(f"{jaeger_url}/api/services", timeout=10)
            assert response.status_code == 200
            
            services_data = response.json()
            assert "data" in services_data
            # Services list might be empty if no traces have been sent yet
            
        except requests.exceptions.RequestException:
            pytest.skip("Jaeger not accessible for services testing")


class TestLogAggregation:
    """Test log aggregation with Loki and Promtail"""
    
    def test_loki_config_valid(self):
        """Test Loki configuration is valid"""
        config_file = "infrastructure/monitoring/loki-config.yml"
        if Path(config_file).exists():
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                
                # Check required sections
                assert "server" in config
                assert "schema_config" in config
                assert "limits_config" in config
                
                # Check server config
                server = config["server"]
                assert "http_listen_port" in server
                assert server["http_listen_port"] == 3100
    
    def test_promtail_config_valid(self):
        """Test Promtail configuration is valid"""
        config_file = "infrastructure/monitoring/promtail-config.yml"
        if Path(config_file).exists():
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                
                # Check required sections
                assert "server" in config
                assert "clients" in config
                assert "scrape_configs" in config
                
                # Check clients point to Loki
                clients = config["clients"]
                assert len(clients) > 0
                assert "loki" in clients[0]["url"]
                
                # Check scrape configs
                scrape_configs = config["scrape_configs"]
                assert len(scrape_configs) > 0
                
                # Check for container logs scraping
                job_names = [job["job_name"] for job in scrape_configs]
                assert "containers" in job_names


class TestPerformanceMetrics:
    """Test performance monitoring and metrics collection"""
    
    def test_node_exporter_metrics(self):
        """Test node exporter metrics are available"""
        try:
            response = requests.get("http://localhost:9100/metrics", timeout=10)
            assert response.status_code == 200
            
            metrics_text = response.text
            
            # Check for essential system metrics
            essential_metrics = [
                "node_cpu_seconds_total",
                "node_memory_MemTotal_bytes",
                "node_filesystem_size_bytes",
                "node_network_receive_bytes_total"
            ]
            
            for metric in essential_metrics:
                assert metric in metrics_text
                
        except requests.exceptions.RequestException:
            pytest.skip("Node exporter not accessible")
    
    def test_cadvisor_metrics(self):
        """Test cAdvisor container metrics are available"""
        try:
            response = requests.get("http://localhost:8080/metrics", timeout=10)
            assert response.status_code == 200
            
            metrics_text = response.text
            
            # Check for container metrics
            container_metrics = [
                "container_cpu_usage_seconds_total",
                "container_memory_usage_bytes",
                "container_network_receive_bytes_total"
            ]
            
            for metric in container_metrics:
                assert metric in metrics_text
                
        except requests.exceptions.RequestException:
            pytest.skip("cAdvisor not accessible")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])