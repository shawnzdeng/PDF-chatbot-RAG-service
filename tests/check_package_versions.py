#!/usr/bin/env python3
"""
Script to check latest compatible package versions for Python 3.10.18
"""
import subprocess
import sys
import json
from packaging import version
import requests

def get_package_info(package_name):
    """Get package information from PyPI"""
    try:
        response = requests.get(f"https://pypi.org/pypi/{package_name}/json")
        if response.status_code == 200:
            data = response.json()
            latest_version = data['info']['version']
            
            # Get Python version compatibility
            python_requires = data['info'].get('requires_python', '')
            
            return {
                'name': package_name,
                'latest_version': latest_version,
                'python_requires': python_requires,
                'summary': data['info']['summary']
            }
    except Exception as e:
        print(f"Error fetching info for {package_name}: {e}")
        return None

def check_python_compatibility(python_requires, target_python="3.10.18"):
    """Check if target Python version is compatible with package requirements"""
    if not python_requires:
        return True, "No specific requirement"
    
    try:
        # This is a simplified check - in reality, you'd want to use packaging.specifiers
        if ">=" in python_requires:
            min_version = python_requires.split(">=")[1].strip().split(",")[0]
            return version.parse(target_python) >= version.parse(min_version), f"Requires >= {min_version}"
        elif ">" in python_requires:
            min_version = python_requires.split(">")[1].strip().split(",")[0]
            return version.parse(target_python) > version.parse(min_version), f"Requires > {min_version}"
        
        return True, python_requires
    except:
        return True, f"Could not parse: {python_requires}"

# Packages from requirements-cpu.txt
packages = [
    'langchain',
    'langchain-openai', 
    'langchain-core',
    'openai',
    'qdrant-client',
    'streamlit',
    'python-dotenv',
    'torch',
    'transformers',
    'sentence-transformers',
    'pytest',
    'pandas',
    'numpy',
    'psutil',
    'httpx'
]

print("Checking package versions for Python 3.10.18 compatibility...")
print("="*80)

results = []
for package in packages:
    print(f"Checking {package}...")
    info = get_package_info(package)
    if info:
        compatible, note = check_python_compatibility(info['python_requires'])
        results.append({
            **info,
            'compatible': compatible,
            'compatibility_note': note
        })

print("\nRESULTS:")
print("="*80)
for result in results:
    status = "✅" if result['compatible'] else "❌"
    print(f"{status} {result['name']}: {result['latest_version']}")
    print(f"   Python req: {result['python_requires'] or 'None specified'}")
    print(f"   Note: {result['compatibility_note']}")
    print()
