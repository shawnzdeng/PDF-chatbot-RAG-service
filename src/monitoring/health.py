"""
Health check and metrics endpoints for the RAG system
Add these routes to your Streamlit app for monitoring
"""

import streamlit as st
import json
from typing import Dict, Any
from src.monitoring.metrics import rag_metrics

def add_health_endpoints():
    """Add health and metrics endpoints to Streamlit app"""
    
    # Add query parameters handling
    query_params = st.query_params
    
    # Health check endpoint
    if query_params.get('health') == 'true':
        health_status = rag_metrics.get_health_status()
        st.json(health_status)
        st.stop()
    
    # Metrics endpoint
    if query_params.get('metrics') == 'true':
        st.text(rag_metrics.get_prometheus_metrics())
        st.stop()
    
    # Ready check endpoint
    if query_params.get('ready') == 'true':
        st.json({"status": "ready", "timestamp": "2025-08-01T00:00:00Z"})
        st.stop()

# Example usage in your main streamlit app:
# Add this to the beginning of your streamlit_app.py after imports:
# from src.monitoring.health import add_health_endpoints
# add_health_endpoints()
