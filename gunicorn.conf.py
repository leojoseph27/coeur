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

# SSL (if needed)
keyfile = None
certfile = None

# Process Naming
proc_name = "coeur_app"

# Server Mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# SSL
ssl_version = "TLS"
cert_reqs = 0  # ssl.CERT_NONE

# Error handling
capture_output = True
enable_stdio_inheritance = True 