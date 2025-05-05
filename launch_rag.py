#!/usr/bin/env python
"""
Launcher script for the Modular RAG System
"""
import os
import sys

# Add the parent directory to Python path so we can import the modular_rag package
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Now we can import modules from modular_rag
from modular_rag.app import build_interface

if __name__ == "__main__":
    # Build and launch the interface
    demo = build_interface()
    demo.launch(show_api=False, share=False)
    
    print("RAG System started successfully!")
