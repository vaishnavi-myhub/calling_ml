# AI Voice Calling — Connected Enterprise Prototype

## Goal
Build a polished, desktop-first light SaaS application for managing real-time inbound and outbound AI calls, with realistic sample data and working navigation across all requested screens.

## What will be built

### Shared application shell
- Fixed 240px light sidebar with the requested navigation, workspace selector, system health, and user profile.
- Minimal top header with contextual page title, global search, notifications, help, and profile controls.
- Collapsible navigation for tablet widths and horizontally scrollable data tables.
- Shared empty/loading/active states, dialogs, notifications, tooltips, and accessible controls.

### Overview
- Greeting, campaign/test-call actions, and six animated KPI cards.
- Interactive call activity chart with period controls and detailed tooltips.
- High-priority live-call monitor with animated waveform, transcript, intelligence badges, and call controls.
- Animated AI Call Flow with directional particles and semantic processing states.
- Recent calls, system health, and quick-action sections.

### Connected product screens
- Dedicated Live Call control center.
- Campaigns with progress and performance charts.
- Contacts / Leads with filters and import/add actions.
- AI Agents with detailed model, language, voice, and performance data.
- Knowledge Base with source categories and retrieval preview.
- Call History with filters and a call-details dialog.
- Analytics with operational, language, funnel, sentiment, and latency views.
- Models & Infrastructure with model/fallback and resource health monitoring.
- Settings with all requested tabs and AI model primary/fallback controls.

### Interaction model
- Navigation links open real routes with route-specific page titles and metadata.
- Tabs, filters, search, selects, call controls, dialogs, menus, and action buttons respond to input.
- Test call and live-call cards open the dedicated monitoring screen.
- Waveforms, call-flow particles, status indicators, transcript entries, chart tooltips, KPI values, and thinking states use subtle motion with reduced-motion support.

## Visual system
- Light-only, airy interface using the supplied neutral, green, blue, purple, orange, amber, and red palette.
- Semantic color roles strictly follow the brief: green health/success, blue communication/STT, purple AI/LLM, orange knowledge/RAG, red failure/end-call only.
- White cards, fine borders, compact radii, restrained shadows, clear type hierarchy, and an 8px spacing rhythm.
- No dark sidebar, black/navy-heavy surfaces, chatbot styling, neon colors, or excessive gradients.

## Technical details
- TanStack Start routes for every major page and shared shell components around the route outlet.
- Reusable primitives for KPIs, charts, badges, status rows, waveforms, transcripts, flow nodes, tables, cards, and page controls.
- Recharts for responsive data visualizations and Lucide line icons throughout.
- CSS design tokens in `src/styles.css`; components use semantic Tailwind utilities rather than hardcoded colors.
- Realistic static demonstration data only; no database or external calling provider is required for this prototype.
- Unique metadata for every content route.

## Verification
- Check every route and primary interaction in the running preview.
- Visually verify the overview and live-call experiences at 1440px desktop and 1024px tablet widths.
- Confirm navigation, charts, dialogs, call controls, table overflow, animated states, and reduced-motion behavior.
