# broker_url = "redis://127.0.0.1:6379/0"
# broker_url = "redis://circuitree-cache2.t3c6zs.ng.0001.usw2.cache.amazonaws.com:6379"
# broker_url = "valkeys://pranav-circuitree-cache-t3c6zs.serverless.usw2.cache.amazonaws.com:6379"
broker_url = (
    "rediss://circuitree-cache-redis-t3c6zs.serverless.usw2.cache.amazonaws.com:6379"
)

result_backend = broker_url
broker_connection_retry_on_startup = True
worker_prefetch_multiplier = 1
worker_cancel_long_running_tasks_on_connection_loss = True
worker_disable_rate_limits = True
# task_compression = "gzip"
broker_transport_options = {
    "fanout_prefix": True,
    "fanout_patterns": True,
    # "socket_keepalive": True,
}
