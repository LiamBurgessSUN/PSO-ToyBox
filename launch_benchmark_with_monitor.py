#!/usr/bin/env python3
"""
Launcher script for PSO-ToyBox benchmark with background resource monitoring.
This script starts resource monitoring in the background and then runs the benchmark.
"""

import os
import sys
import time
import subprocess
import threading
import signal
from pathlib import Path
from resource_monitor import ResourceMonitor

def signal_handler(signum, frame):
    """Handle interrupt signals."""
    print(f"\nReceived signal {signum}. Shutting down gracefully...")
    sys.exit(0)

def run_benchmark_with_background_monitor(benchmark_script: str = "SAPSO_AGENT/Benchmark/benchmark.py",
                                        monitor_interval: float = 1.0,
                                        output_file: str = None):
    """
    Launch benchmark with background resource monitoring.
    
    Args:
        benchmark_script (str): Path to the benchmark script
        monitor_interval (float): Monitoring interval in seconds
        output_file (str): Optional output file for monitoring data
    """
    print("="*60)
    print("PSO-ToyBox Benchmark with Resource Monitoring")
    print("="*60)
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Initialize resource monitor
    print(f"Initializing resource monitor (interval: {monitor_interval}s)")
    monitor = ResourceMonitor(output_file=output_file, interval=monitor_interval)
    
    # Start monitoring in background
    print("Starting background resource monitoring...")
    monitor.start_monitoring()
    
    # Give monitor a moment to start
    time.sleep(0.5)
    
    try:
        # Run the benchmark
        print(f"\nLaunching benchmark: {benchmark_script}")
        print("-" * 40)
        
        # Set environment variables for the benchmark
        env = os.environ.copy()
        env['PYTHONPATH'] = str(Path.cwd())
        
        # Run benchmark with real-time output
        process = subprocess.Popen(
            [sys.executable, benchmark_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            bufsize=1,
            universal_newlines=True
        )
        
        # Monitor the specific process
        monitor.target_pid = process.pid
        
        # Stream output in real-time
        print("Benchmark Output:")
        print("-" * 40)
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.rstrip())
        
        # Wait for process to complete
        return_code = process.poll()
        
        # Stop monitoring
        print("\n" + "-" * 40)
        print("Benchmark completed. Stopping resource monitoring...")
        monitor.stop_monitoring()
        
        # Print final status
        print(f"\nBenchmark exit code: {return_code}")
        print(f"Resource data saved to: {monitor.output_file}")
        
        if return_code == 0:
            print("✅ Benchmark completed successfully!")
        else:
            print("❌ Benchmark completed with errors!")
        
        return return_code
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Shutting down...")
        monitor.stop_monitoring()
        if 'process' in locals():
            process.terminate()
            process.wait()
        return 1
        
    except Exception as e:
        print(f"\nError running benchmark: {e}")
        monitor.stop_monitoring()
        return 1

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Launch PSO-ToyBox benchmark with background resource monitoring"
    )
    parser.add_argument(
        "--script", 
        default="SAPSO_AGENT/Benchmark/benchmark.py",
        help="Path to benchmark script (default: SAPSO_AGENT/Benchmark/benchmark.py)"
    )
    parser.add_argument(
        "--interval", 
        type=float, 
        default=1.0,
        help="Monitoring interval in seconds (default: 1.0)"
    )
    parser.add_argument(
        "--output", 
        help="Output file for monitoring data (default: auto-generated)"
    )
    parser.add_argument(
        "--monitor-only", 
        action="store_true",
        help="Only start monitoring, don't run benchmark"
    )
    
    args = parser.parse_args()
    
    if args.monitor_only:
        # Monitor-only mode
        print("Starting resource monitoring only...")
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
        return_code = run_benchmark_with_background_monitor(
            benchmark_script=args.script,
            monitor_interval=args.interval,
            output_file=args.output
        )
        sys.exit(return_code)

if __name__ == "__main__":
    main() 