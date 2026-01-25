# Architecture Diagram (Mermaid)

Copy this to [Mermaid Live Editor](https://mermaid.live) to generate PNG:

```mermaid
flowchart TB
    subgraph Input
        U[/"User Query"/]
    end

    subgraph Orchestrator["ORCHESTRATOR (LangGraph)"]
        C["classifier<br/>─────────<br/>Haiku · $0.001"]
        R{Route}
    end

    subgraph Agents["SPECIALIST AGENTS"]
        SA["Scalp Agent<br/>─────────<br/>Sonnet · $0.01"]
        FA["Fallback Agent<br/>─────────<br/>Haiku · $0.002"]
    end

    subgraph Tools["TOOLS"]
        T1["get_stock_quote<br/>(Finnhub API)"]
        T2["calculate_confluence<br/>(V2.1 Scoring)"]
        T3["get_spy_change<br/>(Market Context)"]
    end

    subgraph Observability["OBSERVABILITY"]
        P["Phoenix Tracing<br/>───────────<br/>localhost:6006"]
    end

    U --> C
    C --> R
    R -->|"SCALP_ANALYSIS"| SA
    R -->|"GENERAL"| FA
    SA --> T1
    SA --> T2
    SA --> T3
    T1 -.-> P
    T2 -.-> P
    T3 -.-> P
    SA -.-> P
    FA -.-> P
    C -.-> P

    style C fill:#4ade80,color:#000
    style SA fill:#60a5fa,color:#000
    style FA fill:#4ade80,color:#000
    style P fill:#f472b6,color:#000
    style T1 fill:#fbbf24,color:#000
    style T2 fill:#fbbf24,color:#000
    style T3 fill:#fbbf24,color:#000
```

## Simpler Version (for Medium)

```mermaid
flowchart LR
    A["User Query"] --> B["Orchestrator<br/>(Haiku $0.001)"]
    B --> C{"Route"}
    C -->|Complex| D["Scalp Agent<br/>(Sonnet $0.01)"]
    C -->|Simple| E["Fallback Agent<br/>(Haiku $0.002)"]
    D --> F["Tools"]
    F --> G["Response"]
    E --> G

    style B fill:#4ade80
    style D fill:#60a5fa
    style E fill:#4ade80
```

## Even Simpler (Text-Based for Medium)

```
User Query
    │
    ▼
┌──────────────────────────┐
│  ORCHESTRATOR (Haiku)    │  ← $0.001
│  "Route to right agent"  │
└────────────┬─────────────┘
             │
   ┌─────────┴─────────┐
   ▼                   ▼
┌────────┐      ┌────────┐
│ Scalp  │      │Fallback│
│ Agent  │      │ Agent  │
│(Sonnet)│      │(Haiku) │
│ $0.01  │      │ $0.002 │
└───┬────┘      └────────┘
    │
    ▼
┌────────────────────────┐
│  TOOLS                 │
│  • get_stock_quote     │
│  • calculate_confluence│
│  • get_spy_change      │
└────────────────────────┘
    │
    ▼
┌────────────────────────┐
│  PHOENIX TRACING       │
│  All spans visible     │
└────────────────────────┘
```
