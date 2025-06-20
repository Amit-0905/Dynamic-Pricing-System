from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import asyncio
import aioredis
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Dynamic Pricing API",
    description="Real-time pricing optimization service",
    version="1.0.0"
)

# Pydantic models for request/response
class PricingRequest(BaseModel):
    product_id: str
    current_price: float
    competitor_prices: List[float]
    inventory_level: int
    demand_forecast: Optional[float] = None
    customer_segment: str = "regular"
    time_context: Optional[str] = None

class PricingResponse(BaseModel):
    product_id: str
    recommended_price: float
    confidence_score: float
    price_change_percentage: float
    expected_demand: float
    expected_profit: float
    strategy: str
    timestamp: datetime

@app.post("/pricing/optimize", response_model=PricingResponse)
async def optimize_price(request: PricingRequest):
    """Optimize price for a given product based on market conditions"""
    # Core pricing optimization logic
    avg_competitor_price = sum(request.competitor_prices) / len(request.competitor_prices)
    price_ratio = request.current_price / avg_competitor_price

    # Demand forecasting
    if request.demand_forecast:
        expected_demand = request.demand_forecast
    else:
        base_demand = 1000
        price_effect = -10 * (request.current_price - 50)
        inventory_effect = min(request.inventory_level / 1000, 1) * 100
        expected_demand = max(base_demand + price_effect + inventory_effect, 10)

    # Price optimization
    if price_ratio > 1.1:  # Price too high
        price_adjustment = -0.05
        strategy = "aggressive_discount"
    elif price_ratio < 0.9:  # Price too low
        price_adjustment = 0.03
        strategy = "price_increase"
    elif request.inventory_level > 1500:  # High inventory
        price_adjustment = -0.02
        strategy = "inventory_clearance"
    else:
        price_adjustment = 0.01
        strategy = "optimization"

    recommended_price = request.current_price * (1 + price_adjustment)

    # Calculate expected profit
    unit_cost = 20
    expected_profit = expected_demand * (recommended_price - unit_cost)

    # Confidence score
    confidence_score = 0.85 if request.demand_forecast else 0.70

    return PricingResponse(
        product_id=request.product_id,
        recommended_price=round(recommended_price, 2),
        confidence_score=confidence_score,
        price_change_percentage=price_adjustment * 100,
        expected_demand=expected_demand,
        expected_profit=expected_profit,
        strategy=strategy,
        timestamp=datetime.utcnow()
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
