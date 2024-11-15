from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

def rate_limit_key_generator(request: Request):
    """Generate rate limit key based on IP and endpoint"""
    return f"{get_remote_address(request)}:{request.url.path}"
