# Screenshots

This folder holds the images referenced by the top-level `README.md`. Capture them by running `make demo` and grabbing the relevant Streamlit views from the app and sidebar chat experience.

| File | Source | Suggested capture |
| --- | --- | --- |
| `fleet-overview.png` | `src/ui/app.py` (Fleet Overview, default landing page) | KPI row + risk heatmap |
| `asset-deep-dive.png` | `src/ui/pages/1_Asset_Deep_Dive.py` | DGA trend + health index for a chosen `T-####` |
| `agent-chat.gif` | Sidebar chat panel from `src/ui/chat_panel.py` | Short clip of asking "Show me the top 5 highest-risk transformers and why" |

PNGs should be about 1600px wide; GIFs should stay at or below 5 MB.
