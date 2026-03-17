from pydantic import BaseModel
from typing import Dict, Any


class ProxySchema(BaseModel):
    ip : str
    continent_name : str = None
    country_name: str = None
    region_name: str = None
    latitude: float = None
    longitude: float = None
    asn: str = None
    isp: str = None


class PredictRequest(ProxySchema):
    pass

class PredictResponse(BaseModel):
    prediction_code: int
    detectionMethod: list
    confidence: float
    is_proxy: bool
