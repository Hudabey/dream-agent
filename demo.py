#!/usr/bin/env python3
"""
Launch the Dream Agent demo.

Usage:
    python demo.py
    python demo.py --port 7860 --share
"""

import argparse
from src.demo import create_demo


def main():
    parser = argparse.ArgumentParser(description='Launch Dream Agent Demo')
    parser.add_argument('--port', type=int, default=7860, help='Port to run on')
    parser.add_argument('--share', action='store_true', help='Create public link')
    parser.add_argument('--server_name', type=str, default='0.0.0.0', help='Server name')

    args = parser.parse_args()

    print("🧠 Launching Dream Agent Demo...")
    print(f"   Port: {args.port}")
    print(f"   Share: {args.share}")

    demo = create_demo()
    demo.launch(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
