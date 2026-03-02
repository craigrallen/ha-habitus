#!/usr/bin/env python3
"""Standalone web server launcher with visible error output."""
import os
import sys
import traceback

sys.path.insert(0, '/app')

try:
    from habitus.web import start_web
    port = int(os.environ.get('INGRESS_PORT', '8099'))
    print(f"[web] Starting Flask on port {port}", flush=True)
    start_web(port)
except Exception as e:
    print(f"[web] FAILED TO START: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)
