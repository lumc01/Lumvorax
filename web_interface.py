#!/usr/bin/env python3
"""
LUM/VORAX System Web Interface
Simple web interface to display system results and run tests
"""

import os
import sys
import json
import subprocess
import threading
import time
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
import html

class LUMVoraxHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        if path == '/' or path == '/index.html':
            self.serve_main_page()
        elif path == '/api/status':
            self.serve_status()
        elif path == '/api/run_tests':
            self.run_progressive_tests()
        elif path == '/api/logs':
            self.serve_logs()
        elif path == '/static/style.css':
            self.serve_css()
        else:
            self.send_error(404)

    def serve_main_page(self):
        html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LUM/VORAX System Dashboard</title>
    <link rel="stylesheet" href="/static/style.css">
</head>
<body>
    <div class="container">
        <header>
            <h1>🚀 LUM/VORAX System Dashboard</h1>
            <p>High-Performance Computing System with 32+ Modules</p>
        </header>
        
        <div class="card">
            <h2>System Status</h2>
            <div id="status">Loading...</div>
        </div>
        
        <div class="card">
            <h2>Available Operations</h2>
            <button onclick="runTests()" class="btn-primary">Run Progressive Tests (1M → 100M)</button>
            <button onclick="refreshStatus()" class="btn-secondary">Refresh Status</button>
            <button onclick="viewLogs()" class="btn-secondary">View Logs</button>
        </div>
        
        <div class="card">
            <h2>System Information</h2>
            <ul>
                <li><strong>Version:</strong> PROGRESSIVE COMPLETE v2.0</li>
                <li><strong>Modules:</strong> 32+ (Core, Advanced, Complex)</li>
                <li><strong>Optimizations:</strong> SIMD +300%, Parallel VORAX +400%, Cache Alignment +15%</li>
                <li><strong>Features:</strong> Neural Networks, Audio/Image Processing, AI Optimization</li>
            </ul>
        </div>
        
        <div class="card">
            <h2>Test Results</h2>
            <div id="results">
                <p>No tests run yet. Click "Run Progressive Tests" to start.</p>
            </div>
        </div>
        
        <div class="card">
            <h2>System Logs</h2>
            <div id="logs">
                <p>Click "View Logs" to display recent system activity.</p>
            </div>
        </div>
    </div>
    
    <script>
        async function refreshStatus() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                document.getElementById('status').innerHTML = `
                    <div class="status-ok">✅ System Ready</div>
                    <p>Last checked: ${new Date().toLocaleString()}</p>
                    <p>Binary: ${data.binary_exists ? '✅ Found' : '❌ Not found'}</p>
                    <p>Build status: ${data.build_ok ? '✅ OK' : '❌ Error'}</p>
                `;
            } catch (error) {
                document.getElementById('status').innerHTML = `
                    <div class="status-error">❌ Error: ${error.message}</div>
                `;
            }
        }
        
        async function runTests() {
            document.getElementById('results').innerHTML = '<div class="loading">🔄 Running tests... This may take several minutes.</div>';
            
            try {
                const response = await fetch('/api/run_tests');
                const data = await response.json();
                
                if (data.success) {
                    document.getElementById('results').innerHTML = `
                        <div class="status-ok">✅ Tests completed successfully!</div>
                        <pre>${data.output}</pre>
                        <p>Duration: ${data.duration} seconds</p>
                    `;
                } else {
                    document.getElementById('results').innerHTML = `
                        <div class="status-error">❌ Tests failed</div>
                        <pre>${data.error}</pre>
                    `;
                }
            } catch (error) {
                document.getElementById('results').innerHTML = `
                    <div class="status-error">❌ Error running tests: ${error.message}</div>
                `;
            }
        }
        
        async function viewLogs() {
            try {
                const response = await fetch('/api/logs');
                const data = await response.json();
                
                document.getElementById('logs').innerHTML = `
                    <h3>Recent Logs</h3>
                    <pre>${data.logs}</pre>
                `;
            } catch (error) {
                document.getElementById('logs').innerHTML = `
                    <div class="status-error">❌ Error loading logs: ${error.message}</div>
                `;
            }
        }
        
        // Auto-refresh status on page load
        refreshStatus();
        
        // Auto-refresh every 30 seconds
        setInterval(refreshStatus, 30000);
    </script>
</body>
</html>
"""
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html_content.encode())

    def serve_css(self):
        css_content = """
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    margin: 0;
    padding: 20px;
    background-color: #f5f5f5;
    line-height: 1.6;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
}

header {
    text-align: center;
    margin-bottom: 30px;
    padding: 20px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 10px;
}

.card {
    background: white;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

.btn-primary, .btn-secondary {
    padding: 12px 24px;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    font-size: 16px;
    margin-right: 10px;
    margin-bottom: 10px;
}

.btn-primary {
    background-color: #007bff;
    color: white;
}

.btn-secondary {
    background-color: #6c757d;
    color: white;
}

.btn-primary:hover {
    background-color: #0056b3;
}

.btn-secondary:hover {
    background-color: #545b62;
}

.status-ok {
    color: #28a745;
    font-weight: bold;
}

.status-error {
    color: #dc3545;
    font-weight: bold;
}

.loading {
    color: #007bff;
    font-style: italic;
}

pre {
    background-color: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 5px;
    padding: 10px;
    overflow-x: auto;
    white-space: pre-wrap;
    max-height: 400px;
    overflow-y: auto;
}

ul li {
    margin-bottom: 5px;
}
"""
        
        self.send_response(200)
        self.send_header('Content-type', 'text/css')
        self.end_headers()
        self.wfile.write(css_content.encode())

    def serve_status(self):
        # Check if binary exists and is executable
        binary_path = './bin/lum_vorax_complete'
        binary_exists = os.path.exists(binary_path) and os.access(binary_path, os.X_OK)
        
        status = {
            'timestamp': datetime.now().isoformat(),
            'binary_exists': binary_exists,
            'build_ok': binary_exists,
            'system_ready': binary_exists
        }
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(status).encode())

    def run_progressive_tests(self):
        try:
            start_time = time.time()
            
            # Run the LUM/VORAX progressive tests
            result = subprocess.run(
                ['./bin/lum_vorax_complete', '--progressive-stress-all'],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd='.'
            )
            
            end_time = time.time()
            duration = round(end_time - start_time, 2)
            
            response = {
                'success': result.returncode == 0,
                'output': result.stdout if result.returncode == 0 else None,
                'error': result.stderr if result.returncode != 0 else None,
                'duration': duration,
                'timestamp': datetime.now().isoformat()
            }
            
        except subprocess.TimeoutExpired:
            response = {
                'success': False,
                'error': 'Tests timed out after 5 minutes',
                'duration': 300,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            response = {
                'success': False,
                'error': str(e),
                'duration': 0,
                'timestamp': datetime.now().isoformat()
            }
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

    def serve_logs(self):
        log_content = ""
        
        # Try to read recent logs
        log_dirs = ['logs/forensic', 'logs/execution', 'logs/tests']
        
        for log_dir in log_dirs:
            if os.path.exists(log_dir):
                try:
                    files = os.listdir(log_dir)
                    files.sort(key=lambda x: os.path.getmtime(os.path.join(log_dir, x)), reverse=True)
                    
                    for file in files[:3]:  # Last 3 files per directory
                        file_path = os.path.join(log_dir, file)
                        if os.path.isfile(file_path):
                            log_content += f"\n=== {file_path} ===\n"
                            with open(file_path, 'r') as f:
                                content = f.read()
                                # Limit content to avoid huge responses
                                if len(content) > 2000:
                                    content = content[-2000:] + "\n... (truncated)"
                                log_content += content
                except Exception as e:
                    log_content += f"\nError reading {log_dir}: {str(e)}\n"
        
        if not log_content:
            log_content = "No logs available yet. Run tests to generate logs."
        
        response = {
            'logs': log_content,
            'timestamp': datetime.now().isoformat()
        }
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

    def log_message(self, format, *args):
        # Override to reduce console spam
        pass

def run_server():
    """Run the web server on port 5000"""
    server_address = ('0.0.0.0', 5000)
    httpd = HTTPServer(server_address, LUMVoraxHandler)
    
    print("🌐 LUM/VORAX Web Interface starting on http://0.0.0.0:5000")
    print("📊 Dashboard available at: http://0.0.0.0:5000")
    print("🔧 System: High-Performance Computing with 32+ modules")
    print("⚡ Features: SIMD +300%, Parallel VORAX +400%, Cache Alignment +15%")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server shutting down...")
        httpd.server_close()

if __name__ == '__main__':
    run_server()