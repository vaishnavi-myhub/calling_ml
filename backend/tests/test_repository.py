import unittest
from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.db import Base
from backend.models import CallRecord, Campaign
from backend.repositories import SqlRepository


def _isolated_repository() -> SqlRepository:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return SqlRepository(sessionmaker(bind=engine, expire_on_commit=False))


class SqlRepositoryTests(unittest.TestCase):
    def test_campaign_round_trip(self):
        repository = _isolated_repository()
        created_at = datetime.now(timezone.utc).isoformat()
        repository.add_campaign(Campaign(str(uuid4()), "Healthcare follow-up", "Draft", created_at))

        campaigns = repository.campaigns

        self.assertEqual(len(campaigns), 1)
        self.assertEqual(campaigns[0].name, "Healthcare follow-up")
        self.assertEqual(campaigns[0].status, "Draft")

    def test_empty_repository_has_zero_safe_metrics(self):
        repository = _isolated_repository()

        metrics = repository.call_metrics()

        self.assertEqual(metrics["activeCalls"], 0)
        self.assertEqual(metrics["callsToday"], 0)
        self.assertEqual(metrics["averageDuration"], "0m 00s")

    def test_real_call_is_reflected_in_metrics_and_history(self):
        repository = _isolated_repository()
        call = CallRecord(
            id=str(uuid4()),
            caller="Live Call Test",
            agent="Local AI Agent",
            direction="Test",
            language="English",
            duration="01:30",
            lead="Interested",
            sentiment=None,
            time="",
            status="Completed",
        )
        repository.add_call(call, transcript="[]", latency_ms=850)

        metrics = repository.call_metrics()
        calls = repository.calls

        self.assertEqual(metrics["successfulCalls"], 1)
        self.assertEqual(metrics["callsToday"], 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].caller, "Live Call Test")
        self.assertEqual(calls[0].status, "Completed")


if __name__ == "__main__":
    unittest.main()
