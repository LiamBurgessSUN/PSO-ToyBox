#!/usr/bin/env python3
"""
Resource Analysis Script for PSO-ToyBox
Analyzes and visualizes resource monitoring data.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime

def load_monitoring_data(file_path: str):
    """Load monitoring data from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def analyze_resources(data):
    """Analyze resource usage patterns."""
    if not data['data']:
        print("No data to analyze")
        return
    
    # Extract time series data
    timestamps = [d['timestamp'] for d in data['data']]
    start_time = min(timestamps)
    times = [(t - start_time) / 60 for t in timestamps]  # Convert to minutes
    
    # CPU data
    cpu_percent = [d['cpu']['cpu_percent'] for d in data['data']]
    
    # Memory data
    memory_percent = [d['memory']['memory_percent'] for d in data['data']]
    memory_used_gb = [d['memory']['memory_used_gb'] for d in data['data']]
    
    # GPU data (if available)
    gpu_memory_used = []
    if data['data'][0]['gpu'] and 'gpu_memory_allocated_gb' in data['data'][0]['gpu']:
        gpu_memory_used = [d['gpu']['gpu_memory_allocated_gb'] for d in data['data']]
    
    # Process data (if available)
    process_cpu = []
    process_memory = []
    if data['data'][0]['process'] and 'process_cpu_percent' in data['data'][0]['process']:
        process_cpu = [d['process']['process_cpu_percent'] for d in data['data']]
        process_memory = [d['process']['process_memory_gb'] for d in data['data']]
    
    # Calculate statistics
    stats = {
        'duration_minutes': max(times),
        'data_points': len(data['data']),
        'cpu': {
            'avg': np.mean(cpu_percent),
            'max': np.max(cpu_percent),
            'min': np.min(cpu_percent),
            'std': np.std(cpu_percent)
        },
        'memory': {
            'avg_percent': np.mean(memory_percent),
            'max_percent': np.max(memory_percent),
            'avg_used_gb': np.mean(memory_used_gb),
            'max_used_gb': np.max(memory_used_gb)
        }
    }
    
    if gpu_memory_used:
        stats['gpu'] = {
            'avg_memory_gb': np.mean(gpu_memory_used),
            'max_memory_gb': np.max(gpu_memory_used),
            'min_memory_gb': np.min(gpu_memory_used)
        }
    
    if process_cpu:
        stats['process'] = {
            'avg_cpu_percent': np.mean(process_cpu),
            'max_cpu_percent': np.max(process_cpu),
            'avg_memory_gb': np.mean(process_memory),
            'max_memory_gb': np.max(process_memory)
        }
    
    return {
        'times': times,
        'cpu_percent': cpu_percent,
        'memory_percent': memory_percent,
        'memory_used_gb': memory_used_gb,
        'gpu_memory_used': gpu_memory_used,
        'process_cpu': process_cpu,
        'process_memory': process_memory,
        'stats': stats
    }

def plot_resources(analysis_data, output_file=None):
    """Create resource usage plots."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('PSO-ToyBox Resource Usage Analysis', fontsize=16)
    
    times = analysis_data['times']
    
    # CPU Usage
    axes[0, 0].plot(times, analysis_data['cpu_percent'], 'b-', linewidth=1)
    axes[0, 0].set_title('CPU Usage')
    axes[0, 0].set_xlabel('Time (minutes)')
    axes[0, 0].set_ylabel('CPU Usage (%)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 100)
    
    # Memory Usage
    axes[0, 1].plot(times, analysis_data['memory_percent'], 'r-', linewidth=1)
    axes[0, 1].set_title('Memory Usage')
    axes[0, 1].set_xlabel('Time (minutes)')
    axes[0, 1].set_ylabel('Memory Usage (%)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 100)
    
    # Memory Used (GB)
    axes[1, 0].plot(times, analysis_data['memory_used_gb'], 'g-', linewidth=1)
    axes[1, 0].set_title('Memory Used')
    axes[1, 0].set_xlabel('Time (minutes)')
    axes[1, 0].set_ylabel('Memory Used (GB)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # GPU Memory (if available)
    if analysis_data['gpu_memory_used']:
        axes[1, 1].plot(times, analysis_data['gpu_memory_used'], 'purple', linewidth=1)
        axes[1, 1].set_title('GPU Memory Usage')
        axes[1, 1].set_xlabel('Time (minutes)')
        axes[1, 1].set_ylabel('GPU Memory (GB)')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        # Process CPU (if available)
        if analysis_data['process_cpu']:
            axes[1, 1].plot(times, analysis_data['process_cpu'], 'orange', linewidth=1)
            axes[1, 1].set_title('Process CPU Usage')
            axes[1, 1].set_xlabel('Time (minutes)')
            axes[1, 1].set_ylabel('Process CPU (%)')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No GPU/Process data available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Additional Data')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    
    plt.show()

def print_summary(analysis_data):
    """Print a summary of the analysis."""
    stats = analysis_data['stats']
    
    print("\n" + "="*60)
    print("RESOURCE USAGE SUMMARY")
    print("="*60)
    print(f"Duration: {stats['duration_minutes']:.2f} minutes")
    print(f"Data points: {stats['data_points']}")
    
    print(f"\nCPU Usage:")
    print(f"  Average: {stats['cpu']['avg']:.1f}%")
    print(f"  Maximum: {stats['cpu']['max']:.1f}%")
    print(f"  Minimum: {stats['cpu']['min']:.1f}%")
    print(f"  Std Dev: {stats['cpu']['std']:.1f}%")
    
    print(f"\nMemory Usage:")
    print(f"  Average: {stats['memory']['avg_percent']:.1f}% ({stats['memory']['avg_used_gb']:.2f} GB)")
    print(f"  Maximum: {stats['memory']['max_percent']:.1f}% ({stats['memory']['max_used_gb']:.2f} GB)")
    
    if 'gpu' in stats:
        print(f"\nGPU Memory Usage:")
        print(f"  Average: {stats['gpu']['avg_memory_gb']:.2f} GB")
        print(f"  Maximum: {stats['gpu']['max_memory_gb']:.2f} GB")
        print(f"  Minimum: {stats['gpu']['min_memory_gb']:.2f} GB")
    
    if 'process' in stats:
        print(f"\nProcess Usage:")
        print(f"  CPU - Average: {stats['process']['avg_cpu_percent']:.1f}%, Max: {stats['process']['max_cpu_percent']:.1f}%")
        print(f"  Memory - Average: {stats['process']['avg_memory_gb']:.2f} GB, Max: {stats['process']['max_memory_gb']:.2f} GB")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Analyze PSO-ToyBox resource monitoring data")
    parser.add_argument("file", help="Path to monitoring data JSON file")
    parser.add_argument("--plot", help="Output file for plot (optional)")
    parser.add_argument("--no-display", action="store_true", help="Don't display plot")
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from: {args.file}")
    data = load_monitoring_data(args.file)
    
    # Analyze data
    print("Analyzing resource usage...")
    analysis = analyze_resources(data)
    
    # Print summary
    print_summary(analysis)
    
    # Create plot
    if not args.no_display:
        print("Creating plots...")
        plot_resources(analysis, args.plot)

if __name__ == "__main__":
    main() 