#!/usr/bin/env python3
"""
Background Resource Monitor for PSO-ToyBox
Monitors CPU, GPU, memory, and time usage during benchmark execution.
"""

import os
import sys
import time
import json
import psutil
import threading
import subprocess
from datetime import datetime
from pathlib import Path
import argparse
from typing import Dict, List, Any, Optional

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class ResourceMonitor:
    """Background resource monitoring for PSO-ToyBox benchmarks."""
    
    def __init__(self, output_file: Optional[str] = None, interval: float = 1.0):
        """
        Initialize the resource monitor.
        
        Args:
            output_file (str): Path to save monitoring data
            interval (float): Monitoring interval in seconds
        """
        self.interval = interval
        self.monitoring = False
        self.data = []
        self.start_time = None
        
        # Set output file
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_file = f"resource_usage_{timestamp}.json"
        else:
            self.output_file = output_file
            
        # Create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        self.output_file = log_dir / self.output_file
        
        print(f"Resource monitor initialized. Output: {self.output_file}")
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get current system resource information."""
        timestamp = time.time()
        
        # CPU information
        cpu_info = {
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
        }
        
        # Memory information
        memory = psutil.virtual_memory()
        memory_info = {
            'memory_total_gb': memory.total / (1024**3),
            'memory_available_gb': memory.available / (1024**3),
            'memory_used_gb': memory.used / (1024**3),
            'memory_percent': memory.percent,
        }
        
        # Disk information
        disk = psutil.disk_usage('/')
        disk_info = {
            'disk_total_gb': disk.total / (1024**3),
            'disk_used_gb': disk.used / (1024**3),
            'disk_free_gb': disk.free / (1024**3),
            'disk_percent': disk.percent,
        }
        
        # Network information
        network = psutil.net_io_counters()
        network_info = {
            'network_bytes_sent': network.bytes_sent,
            'network_bytes_recv': network.bytes_recv,
        }
        
        # GPU information (if available)
        gpu_info = {}
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                gpu_info = {
                    'gpu_count': torch.cuda.device_count(),
                    'gpu_memory_total_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3),
                    'gpu_memory_allocated_gb': torch.cuda.memory_allocated(0) / (1024**3),
                    'gpu_memory_reserved_gb': torch.cuda.memory_reserved(0) / (1024**3),
                    'gpu_memory_free_gb': (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)) / (1024**3),
                }
            except Exception as e:
                gpu_info = {'error': str(e)}
        
        # Process information (if monitoring specific process)
        process_info = {}
        if hasattr(self, 'target_pid') and self.target_pid:
            try:
                process = psutil.Process(self.target_pid)
                process_info = {
                    'process_cpu_percent': process.cpu_percent(),
                    'process_memory_gb': process.memory_info().rss / (1024**3),
                    'process_memory_percent': process.memory_percent(),
                    'process_threads': process.num_threads(),
                }
            except psutil.NoSuchProcess:
                process_info = {'error': 'Process not found'}
        
        return {
            'timestamp': timestamp,
            'datetime': datetime.fromtimestamp(timestamp).isoformat(),
            'cpu': cpu_info,
            'memory': memory_info,
            'disk': disk_info,
            'network': network_info,
            'gpu': gpu_info,
            'process': process_info,
        }
    
    def monitor_loop(self):
        """Main monitoring loop."""
        print(f"Starting resource monitoring (interval: {self.interval}s)")
        self.start_time = time.time()
        
        while self.monitoring:
            try:
                data_point = self.get_system_info()
                self.data.append(data_point)
                time.sleep(self.interval)
            except KeyboardInterrupt:
                print("\nMonitoring interrupted by user")
                break
            except Exception as e:
                print(f"Error during monitoring: {e}")
                time.sleep(self.interval)
    
    def start_monitoring(self, target_pid: int = None):
        """
        Start background monitoring.
        
        Args:
            target_pid (int): Optional process ID to monitor specifically
        """
        if self.monitoring:
            print("Monitoring already active")
            return
        
        self.target_pid = target_pid
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.monitor_thread.start()
        print(f"Resource monitoring started in background")
    
    def stop_monitoring(self):
        """Stop monitoring and save data."""
        if not self.monitoring:
            print("Monitoring not active")
            return
        
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=5)
        
        # Save data
        self.save_data()
        
        # Print summary
        self.print_summary()
        
        print(f"Resource monitoring stopped. Data saved to: {self.output_file}")
    
    def save_data(self):
        """Save monitoring data to JSON file."""
        output_data = {
            'metadata': {
                'start_time': self.start_time,
                'end_time': time.time(),
                'duration_seconds': time.time() - self.start_time if self.start_time else 0,
                'interval_seconds': self.interval,
                'data_points': len(self.data),
                'system_info': {
                    'platform': sys.platform,
                    'python_version': sys.version,
                    'torch_available': TORCH_AVAILABLE,
                }
            },
            'data': self.data
        }
        
        with open(self.output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
    
    def print_summary(self):
        """Print a summary of the monitoring results."""
        if not self.data:
            print("No data collected")
            return
        
        print("\n" + "="*60)
        print("RESOURCE MONITORING SUMMARY")
        print("="*60)
        
        # Time summary
        duration = time.time() - self.start_time if self.start_time else 0
        print(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        print(f"Data points: {len(self.data)}")
        
        # CPU summary
        cpu_percentages = [d['cpu']['cpu_percent'] for d in self.data]
        print(f"CPU Usage - Avg: {sum(cpu_percentages)/len(cpu_percentages):.1f}%, "
              f"Max: {max(cpu_percentages):.1f}%, Min: {min(cpu_percentages):.1f}%")
        
        # Memory summary
        memory_percentages = [d['memory']['memory_percent'] for d in self.data]
        print(f"Memory Usage - Avg: {sum(memory_percentages)/len(memory_percentages):.1f}%, "
              f"Max: {max(memory_percentages):.1f}%, Min: {min(memory_percentages):.1f}%")
        
        # GPU summary (if available)
        if TORCH_AVAILABLE and torch.cuda.is_available() and self.data[0]['gpu']:
            gpu_memory_used = [d['gpu']['gpu_memory_allocated_gb'] for d in self.data if 'gpu_memory_allocated_gb' in d['gpu']]
            if gpu_memory_used:
                print(f"GPU Memory - Avg: {sum(gpu_memory_used)/len(gpu_memory_used):.2f} GB, "
                      f"Max: {max(gpu_memory_used):.2f} GB, Min: {min(gpu_memory_used):.2f} GB")
        
        print("="*60)

def run_benchmark_with_monitoring(benchmark_script: str, monitor_interval: float = 1.0):
    """
    Run benchmark with background resource monitoring.
    
    Args:
        benchmark_script (str): Path to the benchmark script
        monitor_interval (float): Monitoring interval in seconds
    """
    # Initialize monitor
    monitor = ResourceMonitor(interval=monitor_interval)
    
    try:
        # Start monitoring
        monitor.start_monitoring()
        
        # Run benchmark
        print(f"Starting benchmark: {benchmark_script}")
        process = subprocess.Popen([sys.executable, benchmark_script], 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE,
                                 text=True)
        
        # Monitor the specific process
        monitor.target_pid = process.pid
        
        # Wait for benchmark to complete
        stdout, stderr = process.communicate()
        
        # Stop monitoring
        monitor.stop_monitoring()
        
        # Print benchmark output
        if stdout:
            print("\nBenchmark Output:")
            print(stdout)
        
        if stderr:
            print("\nBenchmark Errors:")
            print(stderr)
        
        print(f"\nBenchmark completed with exit code: {process.returncode}")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        monitor.stop_monitoring()
        if 'process' in locals():
            process.terminate()
    except Exception as e:
        print(f"Error running benchmark: {e}")
        monitor.stop_monitoring()

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Resource Monitor for PSO-ToyBox")
    parser.add_argument("--script", default="SAPSO_AGENT/Benchmark/benchmark.py",
                       help="Path to benchmark script")
    parser.add_argument("--interval", type=float, default=1.0,
                       help="Monitoring interval in seconds")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--monitor-only", action="store_true",
                       help="Only monitor, don't run benchmark")
    
    args = parser.parse_args()
    
    if args.monitor_only:
        # Monitor-only mode
        monitor = ResourceMonitor(output_file=args.output, interval=args.interval)
        try:
            monitor.start_monitoring()
            print("Monitoring active. Press Ctrl+C to stop.")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            monitor.stop_monitoring()
    else:
        # Run benchmark with monitoring
        run_benchmark_with_monitoring(args.script, args.interval)

if __name__ == "__main__":
    main() 