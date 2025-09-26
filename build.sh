#!/bin/bash

# Install Node.js dependencies
npm install

# Install Python and pip
apt-get update
apt-get install -y python3 python3-pip

# Install Python dependencies
pip3 install -r requirements.txt

# Build Next.js application
npm run build
