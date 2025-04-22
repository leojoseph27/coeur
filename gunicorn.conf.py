import os
import multiprocessing

# Binding
bind = "0.0.0.0:" + os.environ.get("PORT", "10000")

# Worker Settings
workers = int(os.environ.get("WEB_CONCURRENCY", multiprocessing.cpu_count() * 2 + 1))
worker_class = "geventwebsocket.gunicorn.workers.GeventWebSocketWorker"
worker_connections = 1000
timeout = 600

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# SSL (if needed)
keyfile = None
certfile = None

# Process Naming
proc_name = "coeur_app"

# Server Mechanics
preload_app = True
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