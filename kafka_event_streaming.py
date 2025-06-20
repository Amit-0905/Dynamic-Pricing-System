import asyncio
import json
import logging
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
from typing import Dict, Any
import aioredis
from datetime import datetime

logger = logging.getLogger(__name__)

class PricingEventStreamer:
    """Event-driven pricing system using Kafka"""

    def __init__(self, kafka_servers="localhost:9092", redis_url="redis://localhost:6379"):
        self.kafka_servers = kafka_servers
        self.redis_url = redis_url
        self.producer = None
        self.consumers = {}
        self.redis_pool = None

    async def start(self):
        """Initialize Kafka producer and Redis connection"""
        self.producer = AIOKafkaProducer(
            bootstrap_servers=self.kafka_servers,
            value_serializer=lambda x: json.dumps(x).encode('utf-8')
        )
        await self.producer.start()

        self.redis_pool = aioredis.ConnectionPool.from_url(
            self.redis_url, encoding="utf-8", decode_responses=True
        )

        logger.info("Pricing event streamer started")

    async def publish_price_change(self, product_id: str, old_price: float, 
                                 new_price: float, reason: str):
        """Publish price change event"""
        event = {
            "event_type": "price_change",
            "product_id": product_id,
            "old_price": old_price,
            "new_price": new_price,
            "price_change_pct": ((new_price - old_price) / old_price) * 100,
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat()
        }

        await self.producer.send("pricing.events", event)
        logger.info(f"Published price change event for {product_id}")

    async def handle_inventory_updates(self, event: Dict[str, Any]):
        """Handle inventory level updates"""
        product_id = event.get("product_id")
        inventory_level = event.get("inventory_level")

        # Update Redis cache
        redis = aioredis.Redis(connection_pool=self.redis_pool)
        await redis.setex(f"inventory:{product_id}", 3600, inventory_level)

        # Trigger pricing re-evaluation if inventory is critically low/high
        if inventory_level < 100:
            await self.trigger_pricing_update(product_id, "low_inventory")
        elif inventory_level > 2000:
            await self.trigger_pricing_update(product_id, "high_inventory")

    async def trigger_pricing_update(self, product_id: str, trigger_reason: str):
        """Trigger a pricing update event"""
        event = {
            "event_type": "pricing_trigger",
            "product_id": product_id,
            "trigger_reason": trigger_reason,
            "timestamp": datetime.utcnow().isoformat()
        }

        await self.producer.send("pricing.triggers", event)
        logger.info(f"Triggered pricing update for {product_id}: {trigger_reason}")
