#!/usr/bin/env python3
"""
Monitor training progress
"""

import time
import os
from pathlib import Path

def monitor_training():
    log_file = Path("output/core_images_training/training.log")
    
    if not log_file.exists():
        print("Training log file not found!")
        return
    
    print("Monitoring training progress...")
    print("Press Ctrl+C to stop monitoring")
    
    last_size = 0
    
    try:
        while True:
            if log_file.exists():
                current_size = log_file.stat().st_size
                
                if current_size > last_size:
                    # Read new content
                    with open(log_file, 'r') as f:
                        f.seek(last_size)
                        new_content = f.read()
                        if new_content.strip():
                            print(new_content.strip())
                    
                    last_size = current_size
            
            time.sleep(5)  # Check every 5 seconds
            
    except KeyboardInterrupt:
        print("\nStopped monitoring.")

if __name__ == '__main__':
    monitor_training() 