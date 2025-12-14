#!/bin/bash
echo "Installing ELENA AI for Termux..."
pkg update -y && pkg upgrade -y
pkg install -y python git wget
pip install requests
echo "Installation complete!"
echo "Run: python elena.py"
