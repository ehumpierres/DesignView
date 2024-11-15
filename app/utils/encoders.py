from fastapi.encoders import jsonable_encoder
from typing import Any, Dict

def safe_json_encode(obj: Any, max_depth: int = 10) -> Any:
    """
    Safely encode objects to JSON with recursion protection
    """
    def _encode(obj: Any, current_depth: int = 0) -> Any:
        if current_depth > max_depth:
            return str(obj)
            
        if isinstance(obj, dict):
            return {
                str(k): _encode(v, current_depth + 1)
                for k, v in obj.items()
            }
        elif isinstance(obj, (list, tuple)):
            return [_encode(item, current_depth + 1) for item in obj]
        elif hasattr(obj, 'dict') and callable(getattr(obj, 'dict')):
            return _encode(obj.dict(), current_depth + 1)
        else:
            try:
                return jsonable_encoder(obj)
            except:
                return str(obj)
                
    return _encode(obj) 