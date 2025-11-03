
import json
import http.server
import socketserver
from pathlib import Path

class DemoHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(Path(__file__).parent), **kwargs)
    
    def do_GET(self):
        if self.path == '/':
            self.path = '/demo.html'
        elif self.path == '/api/cameras':
            self.send_json_response('cameras.json')
            return
        elif self.path == '/api/alerts':
            self.send_json_response('sample_alerts.json')
            return
        elif self.path == '/api/incidents':
            self.send_json_response('sample_incidents.json')
            return
        elif self.path == '/api/evidence':
            self.send_json_response('sample_evidence.json')
            return
        
        super().do_GET()
    
    def send_json_response(self, filename):
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())
        except FileNotFoundError:
            self.send_error(404)

if __name__ == "__main__":
    import sys
    PORT = 8081
    # Try different ports if 8081 is busy
    for port in range(8081, 8090):
        try:
            with socketserver.TCPServer(("", port), DemoHandler) as httpd:
                print(f"Demo server running at http://localhost:{port}")
                httpd.serve_forever()
                break
        except OSError:
            continue
    else:
        print("Could not find available port")
        sys.exit(1)
