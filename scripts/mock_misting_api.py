# scripts/mock_misting_api.py
"""Simulated IoT Misting Cannon REST API endpoint for Hackathon Demo.
Listens on port 5000 and logs actuation commands.
Run with: python scripts/mock_misting_api.py
"""
import http.server
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class MistingHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)
        try:
            payload = json.loads(post_data.decode('utf-8'))
            cannon_id = payload.get("cannon_id", "unknown")
            state = payload.get("state", "UNKNOWN")
            logging.info(f"💦 [HARDWARE SIGNAL] Cannon '{cannon_id}' switched -> {state}")
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "success", "actuated": state}).encode('utf-8'))
        except Exception as e:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(str(e).encode('utf-8'))

    def log_message(self, format, *args):
        pass  # suppress default noisy server logs

if __name__ == "__main__":
    server = http.server.HTTPServer(('127.0.0.1', 5000), MistingHandler)
    print("=" * 60)
    print("🚀 IoT MIST CANNON HARDWARE CONTROLLER LISTENING ON http://127.0.0.1:5000")
    print("=" * 60)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping simulated IoT controller.")
