#!/usr/bin/env python3
"""
Simple HTTP server to serve the Vietnamese Poem Generator frontend
"""
import http.server
import socketserver
import os
import webbrowser
from pathlib import Path

PORT = 3000


class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Add CORS headers for API communication
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        # Handle preflight requests for CORS
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


def main():
    # Change to the directory containing the frontend files
    frontend_dir = Path(__file__).parent
    os.chdir(frontend_dir)

    # Check if index.html exists
    if not Path("index.html").exists():
        print("❌ Error: index.html not found!")
        print(f"Please make sure index.html is in: {frontend_dir}")
        return

    # Create server
    with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
        print("🌸 Vietnamese Poem Generator Frontend Server")
        print("=" * 50)
        print(f"🌐 Server running at: http://localhost:{PORT}")
        print(f"📁 Serving files from: {frontend_dir}")
        print(f"📄 Main file: index.html")
        print("=" * 50)
        print("💡 Make sure your FastAPI backend is running on port 8000")
        print("🛑 Press Ctrl+C to stop the server")
        print()

        # Open browser automatically
        try:
            webbrowser.open(f"http://localhost:{PORT}")
            print("✅ Browser opened automatically!")
        except Exception as e:
            print(f"⚠️  Could not open browser automatically: {e}")
            print(f"🌐 Please open your browser and go to: http://localhost:{PORT}")

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Server stopped.")
        except Exception as e:
            print(f"\n❌ Server error: {e}")


if __name__ == "__main__":
    main()
