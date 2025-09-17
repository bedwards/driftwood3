websockets_kwargs = dict(
    ping_interval=30,     # Send ping every n seconds
    ping_timeout=300,     # Wait n seconds for pong
    close_timeout=15,     # Time to wait for clean close
    max_size=1048576,     # n MB message limit
    max_queue=32,         # Limit queued messages
    compression=None,     # Disable compression for lower latency
)
