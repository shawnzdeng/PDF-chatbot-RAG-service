"""
Separate HTTP server for Prometheus metrics
This runs alongside Streamlit to provide proper metrics endpoint
"""

import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import logging
from typing import Dict, Any

from .metrics import rag_metrics

logger = logging.getLogger(__name__)

class MetricsHandler(BaseHTTPRequestHandler):
    """HTTP handler for metrics endpoints"""
    
    def log_message(self, format, *args):
        """Override to use Python logging instead of stderr"""
        logger.debug(format % args)
    
    def do_GET(self):
        """Handle GET requests"""
        try:
            if self.path == '/metrics':
                # Prometheus metrics endpoint
                self.send_response(200)
                self.send_header('Content-Type', 'text/plain; version=0.0.4')
                self.end_headers()
                
                metrics = rag_metrics.get_prometheus_metrics()
                self.wfile.write(metrics.encode('utf-8'))
                
            elif self.path == '/health':
                # Health check endpoint
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                
                health = rag_metrics.get_health_status()
                self.wfile.write(json.dumps(health).encode('utf-8'))
                
            elif self.path == '/ready':
                # Readiness check endpoint
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                
                ready = {"status": "ready", "timestamp": time.time()}
                self.wfile.write(json.dumps(ready).encode('utf-8'))
                
            else:
                # Not found
                self.send_response(404)
                self.send_header('Content-Type', 'text/plain')
                self.end_headers()
                self.wfile.write(b'Not Found')
                
        except Exception as e:
            logger.error(f"Error handling request {self.path}: {e}")
            self.send_response(500)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(f'Internal Server Error: {str(e)}'.encode('utf-8'))

class MetricsServer:
    """Metrics server that runs in a separate thread"""
    
    def __init__(self, port: int = 8502, host: str = '0.0.0.0'):
        self.port = port
        self.host = host
        self.server = None
        self.thread = None
        
    def start(self):
        """Start the metrics server in a separate thread"""
        try:
            self.server = HTTPServer((self.host, self.port), MetricsHandler)
            self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.thread.start()
            logger.info(f"Metrics server started on {self.host}:{self.port}")
            
            # Test endpoints
            logger.info(f"Metrics endpoint: http://{self.host}:{self.port}/metrics")
            logger.info(f"Health endpoint: http://{self.host}:{self.port}/health")
            logger.info(f"Ready endpoint: http://{self.host}:{self.port}/ready")
            
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
            raise
    
    def stop(self):
        """Stop the metrics server"""
        if self.server:
            self.server.shutdown()
            logger.info("Metrics server stopped")

# Global metrics server instance
metrics_server = MetricsServer()

def start_metrics_server():
    """Start the global metrics server"""
    metrics_server.start()
    return metrics_server
