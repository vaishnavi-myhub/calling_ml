from dataclasses import dataclass


@dataclass(frozen=True)
class CampaignCreate:
    name: str

    @classmethod
    def from_payload(cls, payload: object) -> "CampaignCreate | None":
        if not isinstance(payload, dict):
            return None
        name = payload.get("name")
        if not isinstance(name, str) or not name.strip():
            return None
        return cls(name=name.strip())