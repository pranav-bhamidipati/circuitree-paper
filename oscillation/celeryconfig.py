# import os
# import ssl

# _redis_url = os.environ.get("CELERY_BROKER_URL", "")
# if _redis_url:
#     broker_use_ssl = ssl.CERT_NONE
#     redis_backend_use_ssl = {"ssl_cert_reqs": ssl.CERT_NONE}
# else:
#     # Some apps use redis:// url, no SSL needed
#     _redis_url = os.environ["CELERY_BROKER_URL_INTERNAL"]

# celery_broker_url = _redis_url
# celery_result_backend = _redis_url

broker_url = "redis://127.0.0.1:6379/0"
result_backend = broker_url

broker_connection_retry_on_startup = True
worker_prefetch_multiplier = 1
worker_cancel_long_running_tasks_on_connection_loss = True
# task_compression = "gzip"
broker_transport_options = {
    "fanout_prefix": True,
    "fanout_patterns": True,
    # "socket_keepalive": True,
}
