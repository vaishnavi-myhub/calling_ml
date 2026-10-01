# Aura Voice AI

Create a production-quality enterprise SaaS web application UI for:

"AI Voice Calling"

A modern local-AI voice calling platform for managing inbound and outbound AI-powered phone conversations.

IMPORTANT:

Do NOT create a generic dark AI dashboard.

Do NOT use a black/navy-heavy interface.

Do NOT use the typical purple-only AI color palette.

Do NOT make the UI look like a chatbot.

The product must look like a premium, modern enterprise SaaS platform with a LIGHT, CLEAN, AIRY visual system.

==================================================

1. VISUAL DESIGN DIRECTION

==================================================

Overall visual style:

- Premium enterprise SaaS

- Clean

- Minimal

- Professional

- Modern

- Spacious

- Highly readable

- Production-ready

- Trustworthy

- Technical but approachable

Primary theme:

LIGHT MODE ONLY.

Main background:

#F7FAF8

Card background:

#FFFFFF

Primary brand color:

#12A66A

Dark green:

#087A4B

Light green:

#E8F8F0

Mint:

#DDF7EA

Secondary accent:

#4F7CFF

Soft blue:

#EAF1FF

Purple accent:

#7C5CFC

Soft purple:

#F0ECFF

Warning:

#F59E0B

Soft warning:

#FFF4D8

Error:

#EF4444

Soft error:

#FFE8E8

Main text:

#17352A

Secondary text:

#63756D

Border:

#E3ECE7

Do not overuse colors.

GREEN should clearly represent:

- Active

- Online

- Healthy

- Success

- Completed

- Connected

RED should ONLY represent:

- Failed

- Error

- End Call

- Critical status

BLUE should represent:

- Information

- STT

- Communication

- Neutral actions

PURPLE should represent:

- AI / Intelligence

- LLM

- AI Agent

ORANGE should represent:

- Processing

- Warning

- Knowledge/RAG

Use colored icons and subtle tinted backgrounds instead of large solid colored blocks.

==================================================

2. LAYOUT

==================================================

Desktop-first application.

Design for:

1440px × 1024px

Use:

LEFT SIDEBAR

+

TOP HEADER

+

MAIN CONTENT

Sidebar width:

240px

Sidebar background:

#FFFFFF

Very subtle right border.

Do NOT use a dark sidebar.

Sidebar:

Logo:

AI Voice Calling

Logo icon:

simple voice waveform / soundwave icon.

Subtitle:

"Local AI • Real Conversations"

Navigation:

Overview

Live Calls

Campaigns

Contacts / Leads

AI Agents

Knowledge Base

Call History

Analytics

Models & Infrastructure

Settings

Use simple modern line icons.

Active navigation item:

very light green background

green icon

dark green text

rounded 10px

Example:

Overview

with background:

#E8F8F0

At bottom of sidebar:

Workspace selector:

"Healthcare Demo"

System status card:

● System Online

All services running

User profile:

Vaishnavi Reddy

Product Manager

==================================================

3. TOP HEADER

==================================================

White header.

Left:

Current page title.

Center:

Large search field:

"Search calls, contacts, campaigns..."

Right:

Notification icon

Help icon

Profile avatar

User name

Keep header minimal.

==================================================

4. OVERVIEW DASHBOARD

==================================================

Page title:

"Good morning, Vaishnavi 👋"

Subtitle:

"Monitor your AI calling operations and system performance."

Top-right buttons:

+ Create Campaign

Start Test Call

Primary button:

green #12A66A

Secondary button:

white with green border.

==================================================

5. KPI CARDS

==================================================

Create 6 KPI cards in one row.

Cards:

1. Active Calls

12

↑ 20%

2. Calls Today

248

↑ 12%

3. Successful Calls

216

↑ 18%

4. Avg. Duration

4m 32s

↑ 8%

5. Avg. Latency

420ms

↓ 35%

6. Leads Qualified

38

↑ 22%

Each card:

- white background

- thin border

- 12px radius

- subtle shadow

- small colored circular icon

- large number

- comparison indicator

- tiny sparkline chart

Use different VERY LIGHT accent backgrounds:

green

blue

purple

orange

pink

mint

Do NOT use saturated full-color cards.

==================================================

6. MAIN DASHBOARD

==================================================

Create a 3-column responsive layout.

LEFT / CENTER:

Call Activity chart.

RIGHT:

Live Call card.

Below:

Recent Calls

+

Live System Status

+

Quick Actions

==================================================

7. CALL ACTIVITY

==================================================

Card title:

"Call Activity"

Controls:

Last 24 hours

Last 7 days

Last 30 days

Create a beautiful smooth line/area chart.

Metrics:

Answered

Completed

Failed

Green:

Answered

Blue:

Completed

Red:

Failed

Use subtle gradients only inside chart areas.

Add hover tooltip.

Example tooltip:

4:00 PM

Answered 142

Completed 118

Failed 12

Chart should feel like a professional analytics product.

==================================================

8. LIVE CALL CARD

==================================================

This is one of the MOST IMPORTANT components.

Create a clean "Live Call" monitoring card.

Top:

● In Progress

Caller:

+91 98765 43210

Duration:

00:02:48

Direction:

Outbound

Language:

Telugu

AI Agent:

Sales Assistant

Then create a LIVE ANIMATED AUDIO WAVEFORM.

The waveform should visually pulse continuously.

Use green as the main waveform color.

When AI speaks:

waveform animates.

When customer speaks:

waveform changes to blue.

Show:

"AI Agent is speaking..."

Below:

Language badge:

Telugu

Sentiment:

Positive

Lead:

Interested

Then real-time transcript:

Customer

"I want to know about tomorrow's appointment."

AI Agent

"Sure. Your appointment is scheduled for tomorrow morning at 10 AM."

Show timestamps.

Bottom controls:

Mute

Pause AI

Transfer

End Call

Add Note

"End Call" must be red.

All other controls should be neutral/green.

==================================================

9. LIVE CALL DETAIL SCREEN

==================================================

When clicking the Live Call card, open a dedicated Live Call screen.

Create a professional real-time call control center.

Header:

Live Call

+91 98765 43210

● Connected

00:04:32

Main center:

LARGE AUDIO WAVEFORM

Animated waveform.

Under waveform:

AI Agent is speaking

or

Customer is speaking

or

Listening...

Use animated state transitions.

Below waveform:

REAL-TIME TRANSCRIPT

Use two speaker styles:

Customer:

white card

AI Agent:

light green card

Add timestamps.

Right side:

CALL INTELLIGENCE

Detected Language

Telugu

Current Intent

Appointment Inquiry

Sentiment

Positive

Lead Status

Interested

Confidence

96%

Retrieved Knowledge

3 documents

Response Latency

420ms

At bottom:

Mute

Pause AI

Transfer

End Call

Add Note

==================================================

10. AI CALL FLOW — VERY IMPORTANT

==================================================

Create a visually impressive animated process flow.

Title:

"AI Call Flow"

Subtitle:

"From user speech to intelligent response — powered by local AI"

Flow:

CALLER

↓

TELEPHONY

↓

AUDIO STREAMING

↓

VAD

↓

STT

↓

RAG

↓

LLM

↓

TTS

↓

AUDIO STREAMING

↓

CALLER

Each stage must be a circular icon node with:

icon

name

model

status

Examples:

Caller

Telephony

SIP / Plivo

Audio Streaming

20–50 ms chunks

VAD

Silero

STT

Faster-Whisper

RAG

Qdrant

LLM

LLaMA 3

TTS

Indic Parler-TTS

Audio Streaming

8 kHz

Caller

Use:

GREEN = healthy/active

BLUE = audio/data flow

PURPLE = AI processing

ORANGE = knowledge retrieval

Animate the flow.

Animation concept:

small glowing particles travel from:

Caller

→ Telephony

→ Streaming

→ VAD

→ STT

→ RAG

→ LLM

→ TTS

→ Caller

The animation should continuously move from left to right.

When the user is speaking:

show green/blue audio pulse.

When the AI is processing:

show purple pulsing around LLM.

When RAG retrieves information:

show orange pulse around RAG.

When TTS is speaking:

show green waveform animation.

Keep animation subtle and professional.

Do NOT make it flashy.

==================================================

11. LIVE SYSTEM STATUS

==================================================

Create a card:

"Live System Status"

Rows:

Telephony

Online

STT

Faster-Whisper

Online

LLM

LLaMA 3

Online

TTS

Indic Parler-TTS

Online

RAG

Qdrant

Online

GPU

Healthy

Redis

Healthy

Vector DB

Healthy

Each row:

colored icon

service name

model name

green status dot

status text

Green means healthy.

==================================================

12. RECENT CALLS

==================================================

Create professional data table.

Columns:

Caller

AI Agent

Direction

Language

Duration

Lead Status

Sentiment

Time

Status

Example:

+91 98765 43210

Sales Assistant

Outbound

Telugu

04:32

Interested

Positive

2 min ago

Completed

+91 87654 32109

Support Agent

Inbound

Hindi

06:17

Contacted

Neutral

5 min ago

Completed

Use small rounded status badges.

==================================================

13. QUICK ACTIONS

==================================================

Create 6 cards:

Create Campaign

Add Contact

Create AI Agent

Add Knowledge

View Analytics

System Settings

Each card:

small colored icon

title

short description

Keep them white with subtle colored backgrounds.

==================================================

14. AI AGENTS PAGE

==================================================

Create:

AI Agents

Button:

+ Create AI Agent

Agent cards:

Sales Assistant

Appointment Assistant

Support Agent

Each card contains:

Avatar

Agent name

Description

Languages

Voice

STT model

LLM

TTS

Status

Calls handled

Success rate

Example:

Languages:

English • Hindi • Telugu

STT:

Faster-Whisper

LLM:

LLaMA 3

TTS:

Indic Parler-TTS

Status:

Active

==================================================

15. CAMPAIGNS PAGE

==================================================

Create campaign management UI.

Header:

Campaigns

Button:

+ Create Campaign

Table/cards:

Campaign Name

AI Agent

Contacts

Completed

Answer Rate

Interested

Scheduled

Status

Include:

Campaign progress bar

Call performance chart

Lead conversion chart

Call outcome chart

==================================================

16. CONTACTS / LEADS

==================================================

Create CRM-style table.

Columns:

Name

Phone

Language

Campaign

Lead Status

Last Call

Sentiment

Next Follow-up

Filters:

Language

Campaign

Lead Status

Sentiment

Date

Buttons:

Import CSV

+ Add Contact

==================================================

17. KNOWLEDGE BASE

==================================================

Create:

Knowledge Base

Button:

+ Add Knowledge

Cards:

Documents

FAQs

Websites

Product Information

Policies

Show:

Documents count

Chunks

Last indexed

Status

Create a "Knowledge Search Preview":

Query:

"What are your appointment timings?"

Retrieved documents:

Document 1

Document 2

Document 3

Show relevance score.

==================================================

18. CALL HISTORY

==================================================

Create professional call history table.

Filters:

Date

Agent

Campaign

Language

Sentiment

Lead Status

Outcome

Columns:

Caller

Agent

Direction

Language

Duration

Sentiment

Intent

Lead Status

Date

Recording

Transcript

Clicking a call opens Call Details.

==================================================

19. ANALYTICS

==================================================

Create advanced analytics dashboard.

KPIs:

Total Calls

Answer Rate

Completion Rate

Average Duration

Average Latency

Lead Conversion

Appointment Rate

Customer Sentiment

Charts:

Calls Over Time

Calls By Language

Calls By Campaign

Call Outcomes

Lead Conversion Funnel

Sentiment Distribution

Latency Performance

Create:

"Language Performance"

Rows:

English

Hindi

Telugu

Other Languages

Metrics:

Calls

Accuracy

Latency

Completion Rate

==================================================

20. MODELS & INFRASTRUCTURE

==================================================

Create a technical monitoring dashboard.

Model cards:

VAD

Silero VAD

STT

Faster-Whisper

LLM

LLaMA 3

TTS

Indic Parler-TTS

RAG

Qdrant

Fallbacks:

STT:

Whisper Large v3

LLM:

Mistral 7B

TTS:

Piper TTS

VAD:

WebRTC VAD

Show:

Latency

GPU Usage

Memory

Requests

Errors

Throughput

Also show:

GPU

CPU

RAM

Redis

Vector DB

WebSocket

Telephony

Use green health indicators.

==================================================

21. SETTINGS

==================================================

Create Settings page.

Tabs:

General

Telephony

AI Models

Languages

Voices

Knowledge Base

Notifications

Security

Users & Roles

AI Models section:

STT

Primary: Faster-Whisper

Fallback: Whisper Large v3

LLM

Primary: LLaMA 3

Fallback: Mistral 7B

TTS

Primary: Indic Parler-TTS

Fallback: Piper TTS

VAD

Primary: Silero VAD

Fallback: WebRTC VAD

==================================================

22. MICRO-INTERACTIONS

==================================================

Use subtle professional animations.

Examples:

1. KPI numbers animate when dashboard loads.

2. Live call waveform continuously animates.

3. System status indicators softly pulse.

4. AI Call Flow nodes animate sequentially.

5. Data flows between nodes using moving particles.

6. Buttons have subtle hover transitions.

7. Cards slightly lift on hover.

8. Tables highlight row on hover.

9. Status badges smoothly transition when state changes.

10. Transcript messages appear with a subtle slide/fade animation.

11. "AI is thinking" uses three subtle animated dots.

12. "AI is speaking" shows a pulsing waveform.

Animations must be:

- smooth

- subtle

- fast

- professional

Do NOT use excessive animations.

==================================================

23. COLOR RULES

==================================================

IMPORTANT COLOR RULE:

The interface should NOT look monochromatic.

Use a controlled multi-accent palette:

PRIMARY:

Green

AI:

Purple

Communication:

Blue

Knowledge/RAG:

Orange

Success:

Green

Warning:

Amber

Error:

Red

Background:

Very light gray/green

Cards:

White

But keep approximately:

70% neutral whites/light backgrounds

15% green

7% blue

5% purple

3% orange/red

Never use large dark backgrounds.

Never use neon colors.

Never use excessive gradients.

==================================================

24. DESIGN SYSTEM

==================================================

Create reusable components.

Components:

Button

Input

Search

Dropdown

Tabs

Badge

Avatar

KPI Card

Chart Card

Call Card

Agent Card

Model Card

Status Indicator

Transcript Message

Waveform

Flow Node

Table

Modal

Toast

Sidebar

Header

Create variants:

Default

Hover

Active

Disabled

Loading

Success

Warning

Error

Use consistent:

8px spacing system

10–14px border radius

1px borders

subtle shadows

consistent typography

==================================================

25. RESPONSIVE

==================================================

Desktop:

1440px

Laptop:

1280px

Tablet:

1024px

On smaller screens:

Sidebar becomes collapsible.

Cards become responsive.

Tables become horizontally scrollable.

Live call controls remain easily accessible.

==================================================

26. FINAL UX GOAL

==================================================

The final product should feel like:

"An enterprise-grade AI voice operations platform."

It should communicate:

Real-time

AI-powered

Multilingual

Local AI

Secure

Scalable

Professional

Reliable

The most visually important feature should be the REAL-TIME AI CALL EXPERIENCE and the ANIMATED AI CALL FLOW.

The UI must look clean enough for:

- Client demonstrations

- Product presentations

- Enterprise users

- Developers

- Operations teams

- Call-center managers

Generate the complete connected prototype with realistic sample data and working navigation between screens.

This project was built with [Lovable](https://lovable.dev).

## Build with Lovable

Continue developing this project in the [Lovable editor](https://lovable.dev/projects/20374d09-c201-4a04-b592-6530c82050dc).

- **Ship faster**: describe what you want to build and Lovable handles the code.
- **Stay in sync**: every change made in Lovable is committed straight to this repository.
- **Full ownership**: this code is yours. Push to `main` on GitHub and your changes sync back into Lovable, ready for your next prompt.

## Development

Prefer working locally? You need Node.js and npm — [install with nvm](https://github.com/nvm-sh/nvm#installing-and-updating).

```sh
git clone <this-repository-url>
cd <repository-name>
npm i
npm run dev
```
