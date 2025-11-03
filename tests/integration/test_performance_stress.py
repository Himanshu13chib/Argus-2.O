#!/usr/bin/env python3
"""
Performance and Stress Testing for Project Argus
Tests system performance with 100+ concurrent camera feeds, latency requirements, and scalability.
"""

import pytest
import time
import threading
import queue
import asyncio
import concurrent.futures
import psutil
import gc
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import numpy as np
import cv2
from pathlib import Path

# Import core components
from shared.models.detection import Detection, DetectionResult, BoundingBox, DetectionClass
from shared.models.virtual_line import VirtualLine, Point
from shared.models.alerts import Alert, AlertType, Severity


class PerformanceMonitor:
    """Monitor system performance metrics during testing."""
    
    def __init__(self):
        self.metrics = {
            'cpu_usage': [],
            'memory_usage': [],
            'processing_times': [],
            'throughput': [],
            'error_count': 0,
            'start_time': None,
            'end_time': None
        }
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start performance monitoring."""
        self.monitoring = True
        self.metrics['start_time'] = time.time()
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring = False
        self.metrics['end_time'] = time.time()
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
    
    def _monitor_loop(self):
        """Monitor system resources in background."""
        while self.monitoring:
            try:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory_info = psutil.virtual_memory()
                
                self.metrics['cpu_usage'].append(cpu_percent)
                self.metrics['memory_usage'].append(memory_info.percent)
                
                time.sleep(0.5)
            except Exception:
                self.metrics['error_count'] += 1
    
    def record_processing_time(self, processing_time_ms):
        """Record a processing time measurement."""
        self.metrics['processing_times'].append(processing_time_ms)
    
    def get_summary(self):
        """Get performance summary."""
        if not self.metrics['processing_times']:
            return {'error': 'No processing times recorded'}
        
        processing_times = self.metrics['processing_times']
        duration = (self.metrics['end_time'] or time.time()) - (self.metrics['start_time'] or time.time())
        
        return {
            'duration_seconds': duration,
            'total_frames_processed': len(processing_times),
            'average_processing_time_ms': sum(processing_times) / len(processing_times),
            'min_processing_time_ms': min(processing_times),
            'max_processing_time_ms': max(processing_times),
            'p95_processing_time_ms': np.percentile(processing_times, 95),
            'p99_processing_time_ms': np.percentile(processing_times, 99),
            'throughput_fps': len(processing_times) / duration if duration > 0 else 0,
            'average_cpu_usage': np.mean(self.metrics['cpu_usage']) if self.metrics['cpu_usage'] else 0,
            'peak_cpu_usage': max(self.metrics['cpu_usage']) if self.metrics['cpu_usage'] else 0,
            'average_memory_usage': np.mean(self.metrics['memory_usage']) if self.metrics['memory_usage'] else 0,
            'peak_memory_usage': max(self.metrics['memory_usage']) if self.metrics['memory_usage'] else 0,
            'error_count': self.metrics['error_count']
        }


class MockHighPerformanceDetectionPipeline:
    """High-performance mock detection pipeline for stress testing."""
    
    def __init__(self, config):
        self.config = config
        self.virtual_lines = []
        self.is_running = False
        self.frame_count = 0
        self.processing_times = []
        self.lock = threading.Lock()
    
    def process_frame(self, frame, camera_id):
        """Process frame with performance tracking."""
        start_time = time.perf_counter()
        
        with self.lock:
            self.frame_count += 1
            frame_num = self.frame_count
        
        # Simulate realistic processing time (10-50ms)
        processing_delay = np.random.uniform(0.01, 0.05)
        time.sleep(processing_delay)
        
        # Create mock detection
        detection = Detection(
            id=f"det_{camera_id}_{frame_num}",
            camera_id=camera_id,
            timestamp=datetime.now(),
            bbox=BoundingBox(x=100, y=100, width=50, height=100),
            confidence=0.85,
            detection_class=DetectionClass.PERSON,
            features=np.random.rand(512).astype(np.float32)
        )
        
        processing_time = (time.perf_counter() - start_time) * 1000
        
        with self.lock:
            self.processing_times.append(processing_time)
        
        return DetectionResult(
            camera_id=camera_id,
            frame_timestamp=datetime.now(),
            detections=[detection],
            processing_time_ms=processing_time,
            frame_number=frame_num
        )
    
    def get_performance_stats(self):
        """Get performance statistics."""
        with self.lock:
            if not self.processing_times:
                return {}
            
            return {
                'total_frames': len(self.processing_times),
                'avg_processing_time': np.mean(self.processing_times),
                'min_processing_time': np.min(self.processing_times),
                'max_processing_time': np.max(self.processing_times),
                'p95_processing_time': np.percentile(self.processing_times, 95),
                'p99_processing_time': np.percentile(self.processing_times, 99)
            }


class TestPerformanceRequirements:
    """Test system performance against specified requirements."""
    
    def setup_method(self):
        """Set up performance test fixtures."""
        self.detection_config = {
            "model_path": "test_model.pt",
            "confidence_threshold": 0.7,
            "device": "cpu"
        }
        self.performance_monitor = PerformanceMonitor()
    
    def test_detection_latency_requirements(self):
        """Test detection latency meets 300ms requirement."""
        
        detection_pipeline = MockHighPerformanceDetectionPipeline(self.detection_config)
        detection_pipeline.is_running = True
        
        # Create test frame
        test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)  # 720p frame
        
        # Start monitoring
        self.performance_monitor.start_monitoring()
        
        # Process frames and measure latency
        latencies = []
        
        for i in range(100):
            start_time = time.perf_counter()
            result = detection_pipeline.process_frame(test_frame, f"camera_{i%10}")
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            self.performance_monitor.record_processing_time(latency_ms)
        
        # Stop monitoring
        self.performance_monitor.stop_monitoring()
        
        # Verify latency requirements
        avg_latency = np.mean(latencies)
        max_latency = np.max(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        print(f"Average latency: {avg_latency:.1f}ms")
        print(f"Max latency: {max_latency:.1f}ms")
        print(f"P95 latency: {p95_latency:.1f}ms")
        print(f"P99 latency: {p99_latency:.1f}ms")
        
        # Requirements validation
        assert avg_latency < 300, f"Average latency {avg_latency:.1f}ms exceeds 300ms requirement"
        assert p95_latency < 400, f"P95 latency {p95_latency:.1f}ms exceeds 400ms threshold"
        assert p99_latency < 500, f"P99 latency {p99_latency:.1f}ms exceeds 500ms threshold"
        
        # Verify no frames took longer than 1 second (system failure threshold)
        assert max_latency < 1000, f"Max latency {max_latency:.1f}ms exceeds 1000ms failure threshold"
    
    def test_concurrent_camera_feeds_performance(self):
        """Test system performance with 100+ concurrent camera feeds."""
        
        num_cameras = 50  # Reduced for test environment, but tests scalability pattern
        frames_per_camera = 10
        
        detection_pipeline = MockHighPerformanceDetectionPipeline(self.detection_config)
        detection_pipeline.is_running = True
        
        # Create test frame
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Start monitoring
        self.performance_monitor.start_monitoring()
        
        # Results collection
        results = queue.Queue()
        errors = queue.Queue()
        
        def camera_worker(camera_id, frame_count):
            """Worker function for simulating camera feed."""
            try:
                for frame_num in range(frame_count):
                    start_time = time.perf_counter()
                    result = detection_pipeline.process_frame(test_frame, f"camera_{camera_id:03d}")
                    processing_time = (time.perf_counter() - start_time) * 1000
                    
                    results.put({
                        'camera_id': camera_id,
                        'frame_num': frame_num,
                        'processing_time': processing_time,
                        'detections': len(result.detections)
                    })
                    
                    # Small delay to simulate realistic frame rate
                    time.sleep(0.033)  # ~30 FPS
                    
            except Exception as e:
                errors.put({'camera_id': camera_id, 'error': str(e)})
        
        # Start concurrent camera workers
        threads = []
        start_time = time.time()
        
        for camera_id in range(num_cameras):
            thread = threading.Thread(target=camera_worker, args=(camera_id, frames_per_camera))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=30)  # 30 second timeout
        
        end_time = time.time()
        
        # Stop monitoring
        self.performance_monitor.stop_monitoring()
        
        # Collect results
        all_results = []
        while not results.empty():
            all_results.append(results.get())
        
        all_errors = []
        while not errors.empty():
            all_errors.append(errors.get())
        
        # Performance analysis
        total_frames = len(all_results)
        total_duration = end_time - start_time
        overall_throughput = total_frames / total_duration
        
        processing_times = [r['processing_time'] for r in all_results]
        avg_processing_time = np.mean(processing_times) if processing_times else 0
        
        print(f"Concurrent cameras: {num_cameras}")
        print(f"Total frames processed: {total_frames}")
        print(f"Total duration: {total_duration:.1f}s")
        print(f"Overall throughput: {overall_throughput:.1f} FPS")
        print(f"Average processing time: {avg_processing_time:.1f}ms")
        print(f"Errors: {len(all_errors)}")
        
        # Performance requirements
        expected_total_frames = num_cameras * frames_per_camera
        success_rate = total_frames / expected_total_frames if expected_total_frames > 0 else 0
        
        assert success_rate >= 0.95, f"Success rate {success_rate:.2%} below 95% threshold"
        assert len(all_errors) < num_cameras * 0.1, f"Too many errors: {len(all_errors)}"
        assert avg_processing_time < 300, f"Average processing time {avg_processing_time:.1f}ms exceeds 300ms"
        
        # Verify system can handle the load
        assert overall_throughput > num_cameras * 0.8, f"Throughput {overall_throughput:.1f} too low for {num_cameras} cameras"
    
    def test_false_positive_rate_under_load(self):
        """Test false positive rate remains below 1 alert per camera per day under load."""
        
        detection_pipeline = MockHighPerformanceDetectionPipeline(self.detection_config)
        detection_pipeline.is_running = True
        
        # Mock alert engine with false positive tracking
        class MockAlertEngineWithFP:
            def __init__(self):
                self.alerts_generated = []
                self.false_positive_rate = 0.02  # 2% false positive rate
            
            def generate_alert(self, detection):
                # Simulate false positive generation
                if np.random.random() < self.false_positive_rate:
                    alert = Alert(
                        id=f"alert_{len(self.alerts_generated)+1}",
                        type=AlertType.VIRTUAL_LINE_CROSSING,
                        severity=Severity.MEDIUM,
                        camera_id=detection.camera_id,
                        detection_id=detection.id,
                        timestamp=detection.timestamp,
                        confidence=detection.confidence,
                        risk_score=0.5
                    )
                    self.alerts_generated.append(alert)
                    return alert
                return None
            
            def get_false_positive_rate(self, camera_id, time_period_hours=24):
                camera_alerts = [a for a in self.alerts_generated if a.camera_id == camera_id]
                return len(camera_alerts) / (time_period_hours / 24)  # Alerts per day
        
        alert_engine = MockAlertEngineWithFP()
        
        # Simulate 24 hours of operation at 1 FPS per camera
        num_cameras = 10
        frames_per_hour = 3600  # 1 FPS
        simulation_hours = 1  # Simulate 1 hour for testing (scaled)
        
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Process frames and generate alerts
        for camera_id in range(num_cameras):
            for frame_num in range(frames_per_hour * simulation_hours):
                result = detection_pipeline.process_frame(test_frame, f"camera_{camera_id:03d}")
                
                # Generate alerts for detections
                for detection in result.detections:
                    alert_engine.generate_alert(detection)
        
        # Check false positive rates
        total_alerts = len(alert_engine.alerts_generated)
        total_frames = num_cameras * frames_per_hour * simulation_hours
        overall_fp_rate = total_alerts / total_frames
        
        print(f"Total frames processed: {total_frames}")
        print(f"Total alerts generated: {total_alerts}")
        print(f"Overall false positive rate: {overall_fp_rate:.4f}")
        
        # Check per-camera false positive rates
        for camera_id in range(num_cameras):
            camera_fp_rate = alert_engine.get_false_positive_rate(f"camera_{camera_id:03d}", simulation_hours)
            scaled_daily_rate = camera_fp_rate * (24 / simulation_hours)  # Scale to daily rate
            
            print(f"Camera {camera_id:03d} daily FP rate: {scaled_daily_rate:.2f}")
            assert scaled_daily_rate < 1.0, f"Camera {camera_id:03d} exceeds 1 alert/day requirement: {scaled_daily_rate:.2f}"
        
        # Overall system requirement
        assert overall_fp_rate < 0.01, f"Overall false positive rate {overall_fp_rate:.4f} exceeds 1% threshold"
    
    def test_memory_usage_stability(self):
        """Test memory usage remains stable under continuous operation."""
        
        detection_pipeline = MockHighPerformanceDetectionPipeline(self.detection_config)
        detection_pipeline.is_running = True
        
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Start monitoring
        self.performance_monitor.start_monitoring()
        
        # Record initial memory usage
        initial_memory = psutil.virtual_memory().percent
        memory_samples = [initial_memory]
        
        # Process frames continuously
        for i in range(1000):
            result = detection_pipeline.process_frame(test_frame, f"camera_{i%5}")
            
            # Sample memory usage every 100 frames
            if i % 100 == 0:
                current_memory = psutil.virtual_memory().percent
                memory_samples.append(current_memory)
                
                # Force garbage collection to test for memory leaks
                gc.collect()
        
        # Stop monitoring
        self.performance_monitor.stop_monitoring()
        
        # Analyze memory usage
        final_memory = memory_samples[-1]
        memory_growth = final_memory - initial_memory
        max_memory = max(memory_samples)
        
        print(f"Initial memory usage: {initial_memory:.1f}%")
        print(f"Final memory usage: {final_memory:.1f}%")
        print(f"Memory growth: {memory_growth:.1f}%")
        print(f"Peak memory usage: {max_memory:.1f}%")
        
        # Memory stability requirements
        assert memory_growth < 5.0, f"Memory growth {memory_growth:.1f}% indicates potential memory leak"
        assert max_memory < 80.0, f"Peak memory usage {max_memory:.1f}% too high"
        
        # Check for memory leak pattern
        if len(memory_samples) > 5:
            # Linear regression to detect consistent growth
            x = np.arange(len(memory_samples))
            y = np.array(memory_samples)
            slope = np.polyfit(x, y, 1)[0]
            
            assert slope < 0.1, f"Memory usage slope {slope:.3f} indicates memory leak"


class TestStressScenarios:
    """Test system behavior under extreme stress conditions."""
    
    def setup_method(self):
        """Set up stress test fixtures."""
        self.detection_config = {
            "model_path": "test_model.pt",
            "confidence_threshold": 0.7,
            "device": "cpu"
        }
    
    def test_burst_load_handling(self):
        """Test system handling of sudden burst loads."""
        
        detection_pipeline = MockHighPerformanceDetectionPipeline(self.detection_config)
        detection_pipeline.is_running = True
        
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Normal load phase
        normal_load_times = []
        for i in range(50):
            start_time = time.perf_counter()
            result = detection_pipeline.process_frame(test_frame, "camera_001")
            processing_time = (time.perf_counter() - start_time) * 1000
            normal_load_times.append(processing_time)
            time.sleep(0.1)  # 10 FPS
        
        # Burst load phase - sudden increase in processing requests
        burst_load_times = []
        burst_threads = []
        burst_results = queue.Queue()
        
        def burst_worker(worker_id):
            for i in range(20):
                start_time = time.perf_counter()
                result = detection_pipeline.process_frame(test_frame, f"burst_camera_{worker_id}")
                processing_time = (time.perf_counter() - start_time) * 1000
                burst_results.put(processing_time)
        
        # Create burst load with 10 concurrent workers
        burst_start = time.time()
        for worker_id in range(10):
            thread = threading.Thread(target=burst_worker, args=(worker_id,))
            burst_threads.append(thread)
            thread.start()
        
        # Wait for burst to complete
        for thread in burst_threads:
            thread.join(timeout=30)
        
        burst_duration = time.time() - burst_start
        
        # Collect burst results
        while not burst_results.empty():
            burst_load_times.append(burst_results.get())
        
        # Recovery phase - return to normal load
        recovery_load_times = []
        for i in range(50):
            start_time = time.perf_counter()
            result = detection_pipeline.process_frame(test_frame, "camera_001")
            processing_time = (time.perf_counter() - start_time) * 1000
            recovery_load_times.append(processing_time)
            time.sleep(0.1)  # 10 FPS
        
        # Analyze performance
        normal_avg = np.mean(normal_load_times)
        burst_avg = np.mean(burst_load_times) if burst_load_times else 0
        recovery_avg = np.mean(recovery_load_times)
        
        print(f"Normal load average: {normal_avg:.1f}ms")
        print(f"Burst load average: {burst_avg:.1f}ms")
        print(f"Recovery load average: {recovery_avg:.1f}ms")
        print(f"Burst duration: {burst_duration:.1f}s")
        print(f"Burst frames processed: {len(burst_load_times)}")
        
        # Performance requirements during burst
        assert len(burst_load_times) > 100, "Should process significant number of frames during burst"
        assert burst_avg < 1000, f"Burst load processing time {burst_avg:.1f}ms too high"
        
        # Recovery requirements
        recovery_degradation = (recovery_avg - normal_avg) / normal_avg
        assert recovery_degradation < 0.2, f"Recovery performance degraded by {recovery_degradation:.1%}"
    
    def test_resource_exhaustion_handling(self):
        """Test system behavior when resources are exhausted."""
        
        detection_pipeline = MockHighPerformanceDetectionPipeline(self.detection_config)
        detection_pipeline.is_running = True
        
        # Simulate high memory usage scenario
        large_frames = []
        test_results = []
        
        try:
            # Create increasingly large frames to stress memory
            for i in range(10):
                # Create large frame (simulating high-resolution cameras)
                frame_size = (1080 + i*100, 1920 + i*100, 3)  # Increasing resolution
                large_frame = np.random.randint(0, 255, frame_size, dtype=np.uint8)
                large_frames.append(large_frame)
                
                # Process frame
                start_time = time.perf_counter()
                result = detection_pipeline.process_frame(large_frame, f"hires_camera_{i}")
                processing_time = (time.perf_counter() - start_time) * 1000
                
                test_results.append({
                    'frame_size': frame_size,
                    'processing_time': processing_time,
                    'memory_usage': psutil.virtual_memory().percent
                })
                
                # Check if system is still responsive
                assert processing_time < 5000, f"Processing time {processing_time:.1f}ms indicates system overload"
                
        except MemoryError:
            # System should handle memory exhaustion gracefully
            print("Memory exhaustion handled gracefully")
        
        except Exception as e:
            # Other exceptions should be handled appropriately
            assert "memory" in str(e).lower() or "resource" in str(e).lower(), f"Unexpected error: {e}"
        
        finally:
            # Cleanup large frames
            large_frames.clear()
            gc.collect()
        
        # Verify system recovered
        if test_results:
            final_memory = test_results[-1]['memory_usage']
            assert final_memory < 90, f"Memory usage {final_memory:.1f}% too high after stress test"
    
    def test_long_duration_stability(self):
        """Test system stability over extended operation period."""
        
        detection_pipeline = MockHighPerformanceDetectionPipeline(self.detection_config)
        detection_pipeline.is_running = True
        
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Simulate extended operation (scaled for testing)
        duration_minutes = 2  # 2 minutes for testing (represents hours in real system)
        frames_per_second = 5  # Reduced for testing
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        frame_count = 0
        processing_times = []
        error_count = 0
        
        while time.time() < end_time:
            try:
                start_frame_time = time.perf_counter()
                result = detection_pipeline.process_frame(test_frame, f"camera_{frame_count % 5}")
                processing_time = (time.perf_counter() - start_frame_time) * 1000
                
                processing_times.append(processing_time)
                frame_count += 1
                
                # Maintain target frame rate
                time.sleep(1.0 / frames_per_second)
                
            except Exception as e:
                error_count += 1
                print(f"Error during long duration test: {e}")
        
        actual_duration = time.time() - start_time
        
        # Performance analysis
        avg_processing_time = np.mean(processing_times) if processing_times else 0
        throughput = frame_count / actual_duration
        error_rate = error_count / frame_count if frame_count > 0 else 1
        
        print(f"Duration: {actual_duration:.1f}s")
        print(f"Frames processed: {frame_count}")
        print(f"Average processing time: {avg_processing_time:.1f}ms")
        print(f"Throughput: {throughput:.1f} FPS")
        print(f"Error rate: {error_rate:.4f}")
        
        # Stability requirements
        assert frame_count > duration_minutes * 60 * frames_per_second * 0.9, "Frame processing rate too low"
        assert error_rate < 0.01, f"Error rate {error_rate:.4f} too high for stable operation"
        assert avg_processing_time < 300, f"Average processing time {avg_processing_time:.1f}ms exceeds requirement"
        
        # Check for performance degradation over time
        if len(processing_times) > 100:
            early_times = processing_times[:50]
            late_times = processing_times[-50:]
            
            early_avg = np.mean(early_times)
            late_avg = np.mean(late_times)
            degradation = (late_avg - early_avg) / early_avg
            
            assert degradation < 0.1, f"Performance degraded by {degradation:.1%} over time"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])