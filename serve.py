#!/usr/bin/env python3
"""Local dev server that disables directory listing."""

import http.server
import socketserver
import os
import sys

class NoListingHandler(http.server.SimpleHTTPRequestHandler):
    def list_directory(self, path):
        self.send_error(404, "Not Found")
        return None

    def log_message(self, format, *args):
        pass

class ThreadedServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    server = ThreadedServer(("", port), NoListingHandler)
    print(f"Serving at http://localhost:{port}")
    print("Directory listing disabled. Press Ctrl+C to stop.")
    server.serve_forever()
