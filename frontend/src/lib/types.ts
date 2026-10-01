// Shapes returned by the backend's real (non-mock) endpoints. Kept in one
// place so every page that fetches calls/campaigns/overview data agrees on
// what the API actually returns.

export type CallRecordDto = {
  id: string;
  caller: string;
  agent: string;
  direction: string;
  language: string;
  duration: string;
  lead: string;
  sentiment: string | null;
  time: string;
  status: string;
};

export type CampaignDto = {
  id: string;
  name: string;
  status: string;
  createdAt: string;
};

export type ActivityPointDto = {
  time: string;
  answered: number;
  completed: number;
  failed: number;
};

export type SystemServiceDto = {
  name: string;
  model: string;
  status: string;
};

export type OverviewDto = {
  metrics: {
    activeCalls: number;
    callsToday: number;
    successfulCalls: number;
    averageDuration: string;
    averageLatency: string;
    leadsQualified: number;
  };
  activity: ActivityPointDto[];
  recentCalls: CallRecordDto[];
  services: SystemServiceDto[];
};
