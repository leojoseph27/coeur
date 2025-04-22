import os

# Basic Configuration
bind = f"0.0.0.0:{os.environ.get('PORT', '10000')}"
workers = 2
worker_class = "geventwebsocket.gunicorn.workers.GeventWebSocketWorker"
timeout = 120

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Server Mechanics
preload_app = True
daemon = False
pidfile = None

# Process Naming
proc_name = "coeur_app"

# Error handling
capture_output = True
enable_stdio_inheritance = True 