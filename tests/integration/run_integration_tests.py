#!/usr/bin/env python3
"""
Integration Test Runner for Project Argus
Runs comprehensive end-to-end integration tests with proper setup and teardown.
"""

import os
import sys
import subprocess
import time
import json
import tempfile
from pathlib import Path
from datetime import datetime
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class IntegrationTestRunner:
    """Manages integration test execution with proper environment setup."""
    
    def __init__(self, test_config=None):
        self.test_config = test_config or {}
        self.temp_dirs = []
        self.started_services = []
        self.test_results = {}
        
    def setup_test_environment(self):
        """Set up test environment with required services and data."""
        print("🔧 Setting up integration test environment...")
        
        # Create temporary directories
        self.evidence_temp_dir = tempfile.mkdtemp(prefix="argus_evidence_test_")
        self.reports_temp_dir = tempfile.mkdtemp(prefix="argus_reports_test_")
        self.temp_dirs.extend([self.evidence_temp_dir, self.reports_temp_dir])
        
        # Set environment variables for testing
        os.environ.update({
            'ARGUS_TEST_MODE': 'true',
            'ARGUS_EVIDENCE_PATH': self.evidence_temp_dir,
            'ARGUS_REPORTS_PATH': self.reports_temp_dir,
            'ARGUS_DB_URL': 'sqlite:///test_argus.db',
            'ARGUS_LOG_LEVEL': 'INFO'
        })
        
        # Start required test services if needed
        self._start_test_services()
        
        print(f"✅ Test environment ready")
        print(f"   Evidence path: {self.evidence_temp_dir}")
        print(f"   Reports path: {self.reports_temp_dir}")
    
    def _start_test_services(self):
        """Start minimal services required for integration testing."""
        # In a full implementation, this would start test databases, message queues, etc.
        # For now, we'll use mocked services
        
        services_to_start = self.test_config.get('services', [])
        
        for service in services_to_start:
            if service == 'redis':
                self._start_redis_test_instance()
            elif service == 'postgres':
                self._start_postgres_test_instance()
    
    def _start_redis_test_instance(self):
        """Start Redis test instance if available."""
        try:
            # Check if Redis is available
            result = subprocess.run(['redis-server', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                # Start Redis on test port
                redis_process = subprocess.Popen([
                    'redis-server', '--port', '6380', '--daemonize', 'yes'
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                time.sleep(2)  # Wait for startup
                
                # Verify Redis is running
                test_result = subprocess.run([
                    'redis-cli', '-p', '6380', 'ping'
                ], capture_output=True, text=True)
                
                if 'PONG' in test_result.stdout:
                    self.started_services.append(('redis', 6380))
                    os.environ['ARGUS_REDIS_URL'] = 'redis://localhost:6380'
                    print("✅ Redis test instance started on port 6380")
                else:
                    print("⚠️  Redis test instance failed to start")
        except Exception as e:
            print(f"⚠️  Could not start Redis test instance: {e}")
    
    def _start_postgres_test_instance(self):
        """Start PostgreSQL test instance if available."""
        try:
            # Check if PostgreSQL is available
            result = subprocess.run(['pg_ctl', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print("⚠️  PostgreSQL test setup not implemented - using SQLite")
                os.environ['ARGUS_DB_URL'] = f'sqlite:///{self.evidence_temp_dir}/test_argus.db'
        except Exception as e:
            print(f"⚠️  PostgreSQL not available, using SQLite: {e}")
            os.environ['ARGUS_DB_URL'] = f'sqlite:///{self.evidence_temp_dir}/test_argus.db'
    
    def run_test_suite(self, test_pattern=None, verbose=False):
        """Run the integration test suite."""
        print("🧪 Running Project Argus Integration Tests")
        print("=" * 50)
        
        # Define test modules to run
        test_modules = [
            'tests.integration.test_end_to_end_workflows',
            'tests.integration.test_multi_camera_data_flow'
        ]
        
        if test_pattern:
            test_modules = [m for m in test_modules if test_pattern in m]
        
        overall_success = True
        
        for module in test_modules:
            print(f"\n📋 Running {module}...")
            success = self._run_test_module(module, verbose)
            self.test_results[module] = success
            
            if not success:
                overall_success = False
                print(f"❌ {module} FAILED")
            else:
                print(f"✅ {module} PASSED")
        
        return overall_success
    
    def _run_test_module(self, module, verbose=False):
        """Run a specific test module."""
        try:
            # Build pytest command
            cmd = [
                sys.executable, '-m', 'pytest',
                module.replace('.', '/') + '.py',
                '-v' if verbose else '-q',
                '--tb=short',
                '--disable-warnings'
            ]
            
            # Add coverage if requested
            if self.test_config.get('coverage', False):
                cmd.extend(['--cov=edge', '--cov=services', '--cov=shared'])
            
            # Run the test
            result = subprocess.run(
                cmd,
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per module
            )
            
            if verbose or result.returncode != 0:
                print("STDOUT:", result.stdout)
                if result.stderr:
                    print("STDERR:", result.stderr)
            
            return result.returncode == 0
            
        except subprocess.TimeoutExpired:
            print(f"⏰ Test module {module} timed out")
            return False
        except Exception as e:
            print(f"💥 Error running {module}: {e}")
            return False
    
    def run_performance_tests(self):
        """Run performance-specific integration tests."""
        print("\n🚀 Running Performance Integration Tests")
        print("=" * 40)
        
        performance_tests = [
            'tests.integration.test_end_to_end_workflows::TestEndToEndWorkflows::test_system_performance_under_load'
        ]
        
        for test in performance_tests:
            print(f"⏱️  Running {test}...")
            
            start_time = time.time()
            
            cmd = [
                sys.executable, '-m', 'pytest',
                test,
                '-v',
                '--tb=short'
            ]
            
            result = subprocess.run(
                cmd,
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout for performance tests
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                print(f"✅ Performance test passed in {duration:.1f}s")
            else:
                print(f"❌ Performance test failed in {duration:.1f}s")
                print("STDOUT:", result.stdout)
                if result.stderr:
                    print("STDERR:", result.stderr)
    
    def generate_test_report(self):
        """Generate comprehensive test report."""
        print("\n📊 Generating Integration Test Report")
        print("=" * 40)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for success in self.test_results.values() if success)
        failed_tests = total_tests - passed_tests
        
        print(f"Total test modules: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success rate: {(passed_tests/total_tests)*100:.1f}%" if total_tests > 0 else "No tests run")
        
        # Generate detailed report
        report = {
            'timestamp': datetime.now().isoformat(),
            'environment': {
                'evidence_path': self.evidence_temp_dir,
                'reports_path': self.reports_temp_dir,
                'services_started': self.started_services
            },
            'results': self.test_results,
            'summary': {
                'total': total_tests,
                'passed': passed_tests,
                'failed': failed_tests,
                'success_rate': (passed_tests/total_tests)*100 if total_tests > 0 else 0
            }
        }
        
        # Save report
        report_file = Path(self.reports_temp_dir) / 'integration_test_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Detailed report saved to: {report_file}")
        
        return passed_tests == total_tests
    
    def cleanup_test_environment(self):
        """Clean up test environment and resources."""
        print("\n🧹 Cleaning up test environment...")
        
        # Stop started services
        for service_type, port in self.started_services:
            if service_type == 'redis':
                try:
                    subprocess.run(['redis-cli', '-p', str(port), 'shutdown'], 
                                 capture_output=True, timeout=5)
                    print(f"✅ Stopped Redis on port {port}")
                except Exception as e:
                    print(f"⚠️  Error stopping Redis: {e}")
        
        # Clean up temporary directories
        import shutil
        for temp_dir in self.temp_dirs:
            try:
                shutil.rmtree(temp_dir)
                print(f"✅ Cleaned up {temp_dir}")
            except Exception as e:
                print(f"⚠️  Error cleaning up {temp_dir}: {e}")
        
        # Clean up environment variables
        test_env_vars = [k for k in os.environ.keys() if k.startswith('ARGUS_')]
        for var in test_env_vars:
            os.environ.pop(var, None)
        
        print("✅ Test environment cleanup complete")


def main():
    """Main entry point for integration test runner."""
    parser = argparse.ArgumentParser(description='Run Project Argus Integration Tests')
    parser.add_argument('--pattern', '-p', help='Test pattern to match')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--performance', action='store_true', help='Run performance tests')
    parser.add_argument('--coverage', action='store_true', help='Generate coverage report')
    parser.add_argument('--services', nargs='*', default=[], 
                       help='Services to start (redis, postgres)')
    
    args = parser.parse_args()
    
    # Configure test runner
    test_config = {
        'coverage': args.coverage,
        'services': args.services
    }
    
    runner = IntegrationTestRunner(test_config)
    
    try:
        # Setup test environment
        runner.setup_test_environment()
        
        # Run main test suite
        success = runner.run_test_suite(args.pattern, args.verbose)
        
        # Run performance tests if requested
        if args.performance:
            runner.run_performance_tests()
        
        # Generate report
        overall_success = runner.generate_test_report()
        
        # Exit with appropriate code
        sys.exit(0 if overall_success else 1)
        
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)
    finally:
        # Always cleanup
        runner.cleanup_test_environment()


if __name__ == "__main__":
    main()