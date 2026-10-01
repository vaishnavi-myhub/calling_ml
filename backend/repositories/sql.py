"""SQLite-backed repository. Replaces the old hardcoded MemoryRepository:
calls and campaigns are real rows, populated by actual Live Call sessions
and the existing campaign-create flow, and survive a server restart.
"""

from datetime import datetime, timezone

from sqlalchemy.orm import Session, sessionmaker

from ..models import ActivityPoint, CallORM, CallRecord, Campaign, CampaignORM

_ACTIVITY_BUCKETS = ("12 AM", "4 AM", "8 AM", "12 PM", "4 PM", "8 PM", "Now")


def _relative_time(moment: datetime) -> str:
    now = datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    seconds = max(0, int((now - moment).total_seconds()))
    if seconds < 60:
        return "Just now"
    if seconds < 3600:
        return f"{seconds // 60} min ago"
    if seconds < 86400:
        return f"{seconds // 3600} hr ago"
    return f"{seconds // 86400} day ago"


def _call_to_dataclass(row: CallORM) -> CallRecord:
    return CallRecord(
        id=row.id,
        caller=row.caller,
        agent=row.agent,
        direction=row.direction,
        language=row.language,
        duration=row.duration,
        lead=row.lead,
        sentiment=row.sentiment,
        time=_relative_time(row.created_at),
        status=row.status,
    )


class SqlRepository:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    @property
    def calls(self) -> list[CallRecord]:
        with self._session_factory() as session:
            rows = session.query(CallORM).order_by(CallORM.created_at.desc()).limit(200).all()
            return [_call_to_dataclass(row) for row in rows]

    @property
    def campaigns(self) -> list[Campaign]:
        with self._session_factory() as session:
            rows = session.query(CampaignORM).order_by(CampaignORM.created_at.desc()).all()
            return [Campaign(row.id, row.name, row.status, row.created_at) for row in rows]

    def add_campaign(self, campaign: Campaign) -> Campaign:
        with self._session_factory() as session:
            session.add(CampaignORM(id=campaign.id, name=campaign.name, status=campaign.status, created_at=campaign.created_at))
            session.commit()
        return campaign

    def add_call(self, call: CallRecord, transcript: str | None = None, latency_ms: int | None = None) -> CallRecord:
        with self._session_factory() as session:
            session.add(
                CallORM(
                    id=call.id,
                    caller=call.caller,
                    agent=call.agent,
                    direction=call.direction,
                    language=call.language,
                    duration=call.duration,
                    lead=call.lead,
                    sentiment=call.sentiment,
                    status=call.status,
                    transcript=transcript,
                    latency_ms=latency_ms,
                )
            )
            session.commit()
        return call

    def call_metrics(self) -> dict[str, object]:
        """Real aggregates for the Overview page. Zero-safe when there is no history yet."""
        with self._session_factory() as session:
            rows = session.query(CallORM).all()
        total = len(rows)
        completed = [row for row in rows if row.status == "Completed"]
        active = [row for row in rows if row.status == "In Progress"]
        qualified = [row for row in rows if row.lead in {"Interested", "Follow-up"}]

        today = datetime.now(timezone.utc).date()
        calls_today = sum(1 for row in rows if row.created_at.date() == today)

        durations = []
        for row in completed:
            minutes, _, seconds = row.duration.partition(":")
            if minutes.isdigit() and seconds.isdigit():
                durations.append(int(minutes) * 60 + int(seconds))
        average_seconds = round(sum(durations) / len(durations)) if durations else 0

        latencies = [row.latency_ms for row in rows if row.latency_ms is not None]
        average_latency = f"{round(sum(latencies) / len(latencies))}ms" if latencies else "No data yet"

        return {
            "activeCalls": len(active),
            "callsToday": calls_today,
            "successfulCalls": len(completed),
            "averageDuration": f"{average_seconds // 60}m {average_seconds % 60:02d}s" if durations else "0m 00s",
            "averageLatency": average_latency,
            "leadsQualified": len(qualified),
            "totalCalls": total,
        }

    def activity_series(self) -> list[ActivityPoint]:
        """Bucket real calls into the same 7-point shape the chart expects. Empty (all-zero)
        buckets when there is no history yet, instead of fabricated numbers."""
        with self._session_factory() as session:
            rows = session.query(CallORM).all()

        buckets = {label: {"answered": 0, "completed": 0, "failed": 0} for label in _ACTIVITY_BUCKETS}
        for row in rows:
            hour = row.created_at.hour
            index = min(hour // 4, 5)
            label = _ACTIVITY_BUCKETS[index]
            buckets[label]["answered"] += 1
            if row.status == "Completed":
                buckets[label]["completed"] += 1
            elif row.status == "Failed":
                buckets[label]["failed"] += 1

        return [ActivityPoint(label, **buckets[label]) for label in _ACTIVITY_BUCKETS]
