from datetime import datetime, timezone
from uuid import uuid4

from ..models import Campaign
from ..repositories import SqlRepository
from ..schemas import CampaignCreate


class CampaignService:
    def __init__(self, repository: SqlRepository) -> None:
        self.repository = repository

    def create(self, request: CampaignCreate) -> Campaign:
        created_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        return self.repository.add_campaign(Campaign(str(uuid4()), request.name, "Draft", created_at))