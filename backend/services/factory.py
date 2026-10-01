from ..db import SessionFactory
from ..repositories import SqlRepository
from .voice import VoiceService
from .campaigns import CampaignService
from .overview import OverviewService

repository = SqlRepository(SessionFactory)
voice = VoiceService()
campaigns = CampaignService(repository)
overview = OverviewService(repository, voice)
