import os
import json
import glob
from http.server import SimpleHTTPRequestHandler, HTTPServer
import urllib.parse
from pathlib import Path

PORT = 8000
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FRONTEND_DIR = os.path.join(BASE_DIR, "Tools", "viewer_frontend")

class EvalReportHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=FRONTEND_DIR, **kwargs)

    def do_GET(self):
        parsed_path = urllib.parse.urlparse(self.path)
        
        # API endpoint to fetch all reports
        if parsed_path.path == '/api/reports':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            # Handle CORS if needed
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            reports = []
            
            # Search for mujoco_eval_report*.json recursively in logs/
            search_pattern = os.path.join(BASE_DIR, "logs", "**", "*mujoco_eval_report*.json")
            # Also search specifically in checkpoints folders if pattern above misses
            search_pattern2 = os.path.join(BASE_DIR, "IsaacLab_Tasks", "**", "*mujoco_eval_report*.json")
            
            all_files = glob.glob(search_pattern, recursive=True) + glob.glob(search_pattern2, recursive=True)
            
            # Remove duplicates
            all_files = list(set(all_files))
            
            for file_path in all_files:
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        # Ensure it has metadata
                        if "metadata" not in data:
                            # Try to infer it or skip
                            data = {
                                "metadata": {
                                    "checkpoint": "Unknown",
                                    "checkpoint_name": os.path.basename(file_path),
                                    "robot_type": "Unknown",
                                    "timestamp": "Legacy"
                                },
                                "results": data
                            }
                        
                        # Add the file path to metadata for unique identification
                        data["metadata"]["file_path"] = file_path
                        
                        # Generate a nice display name (e.g., "NiceGait6 - 275k")
                        checkpoint_path = data["metadata"].get("checkpoint", "")
                        if checkpoint_path and "checkpoints" in checkpoint_path:
                            parts = checkpoint_path.split(os.sep)
                            try:
                                chk_idx = parts.index("checkpoints")
                                run_name = parts[chk_idx - 1]
                                agent_name = parts[-1].replace(".pt", "").replace("agent_", "")
                                if agent_name.isdigit():
                                    steps = int(agent_name)
                                    if steps >= 1000:
                                        agent_name = f"{steps//1000}k"
                                data["metadata"]["display_name"] = f"{run_name} - {agent_name}"
                            except Exception:
                                data["metadata"]["display_name"] = data["metadata"].get("checkpoint_name", "Unknown")
                        else:
                            data["metadata"]["display_name"] = data["metadata"].get("checkpoint_name", "Unknown")
                        
                        reports.append(data)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    
            # Sort reports by timestamp, descending
            def get_timestamp(r):
                return r.get("metadata", {}).get("timestamp", "")
                
            reports.sort(key=get_timestamp, reverse=True)
            
            response = json.dumps({"reports": reports})
            self.wfile.write(response.encode('utf-8'))
            return
            
        # Serve frontend files
        return super().do_GET()

def main():
    os.makedirs(FRONTEND_DIR, exist_ok=True)
    
    server_address = ('', PORT)
    httpd = HTTPServer(server_address, EvalReportHandler)
    
    print("="*60)
    print(f"🚀 Policy Evaluation Dashboard running!")
    print(f"🔗 Open http://localhost:{PORT} in your browser.")
    print("="*60)
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.server_close()

if __name__ == '__main__':
    main()
