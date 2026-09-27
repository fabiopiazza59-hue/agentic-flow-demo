# Project Brief: FinAdvisor AI Platform

## Executive Summary

**Project Name:** FinAdvisor AI
**Version:** 1.1
**Last Updated:** January 2026
**Target Development Tools:** Kiro IDE / Claude Code
**Validated By:** Simple MVP (Scalp Trading Assistant)

**Vision:** Build an all-in-one agentic AI platform for financial advice that leverages specialized agents, MCP servers, and comprehensive observability to deliver personalized, compliant, and trustworthy financial guidance.

> **Note:** This brief has been updated based on lessons learned from building a working MVP. See "Lessons from MVP" section for practical insights.

---

# PHASE 1: Best Practices & High-Level Design

## 1.1 Architectural Principles

Based on 2025 best practices for agentic AI systems, the platform will adhere to these core principles:

### Decoupled Architecture
- **Microservices-based agent design**: Each agent has a single, well-defined purpose
- **Loose coupling**: Agents communicate through standardized protocols (MCP)
- **Independent deployability**: Each component can be updated without affecting others

### Model-Agnostic Design
- **LLM abstraction layer**: Support multiple providers (Anthropic Claude, OpenAI, AWS Bedrock, local models)
- **Hot-swappable models**: Change underlying models without code changes
- **Cost optimization**: Route requests to appropriate model tiers based on complexity

### Interchangeable Components
- **Plugin architecture**: Tools and skills as pluggable modules
- **MCP-first integration**: All external services exposed via MCP servers
- **Standard interfaces**: Consistent APIs across all agent types

### Observability-First
- **Phoenix tracing from day one**: Every agent interaction traced
- **Evaluation-driven development**: Built-in quality metrics
- **Production monitoring**: Real-time dashboards and alerts

---

## 1.2 Technology Stack (Best-in-Class 2025)

### Core Framework
```
┌─────────────────────────────────────────────────────────────┐
│                    ORCHESTRATION LAYER                       │
│  LangGraph / Strands Agents / Custom Orchestrator           │
├─────────────────────────────────────────────────────────────┤
│                    AGENT FRAMEWORK                           │
│  LlamaIndex Agents / LangChain / AutoGen                    │
├─────────────────────────────────────────────────────────────┤
│                    LLM ABSTRACTION                           │
│  LiteLLM / OpenRouter / Direct API                          │
├─────────────────────────────────────────────────────────────┤
│                    OBSERVABILITY                             │
│  Arize Phoenix (Self-hosted or Cloud)                       │
├─────────────────────────────────────────────────────────────┤
│                    TOOL INTEGRATION                          │
│  Model Context Protocol (MCP) Servers                       │
└─────────────────────────────────────────────────────────────┘
```

### Technology Choices

| Component | Technology | Rationale |
|-----------|------------|-----------|
| **Orchestration** | LangGraph | Cyclic graphs, state management, deterministic control for compliance |
| **Agent Framework** | LlamaIndex + Custom | RAG capabilities, modular agents, financial domain support |
| **LLM Provider** | Claude Sonnet 4 (primary) | Best reasoning, with fallback to GPT-4o, Bedrock |
| **Vector DB** | Qdrant / Weaviate | Open-source, scalable, self-hostable |
| **Memory** | Redis + PostgreSQL | Short-term (Redis) + Long-term (PG with pgvector) |
| **Observability** | Arize Phoenix | Open-source, OpenTelemetry-native, agent-aware |
| **API Layer** | FastAPI | Async, high-performance, OpenAPI docs |
| **MCP Runtime** | Python (FastMCP) | Official SDK, mature ecosystem |
| **Infrastructure** | Docker + K8s | Portable, scalable, cloud-agnostic |

---

## 1.3 High-Level Design (HLD)

### System Architecture Diagram

```
┌────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT LAYER                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │   Web App    │  │  Mobile App  │  │   CLI/API    │  │   Webhooks   │   │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘   │
└────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                            API GATEWAY LAYER                                │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  FastAPI Gateway │ Auth (OAuth 2.1) │ Rate Limiting │ Request Router │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                         ORCHESTRATION LAYER                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                    SUPERVISOR AGENT (LangGraph)                      │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │  │
│  │  │   Intent    │  │    Task     │  │   Response  │                  │  │
│  │  │ Classifier  │  │  Delegator  │  │  Aggregator │                  │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                  │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          ▼                           ▼                           ▼
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│  SPECIALIST       │      │  SPECIALIST       │      │  SPECIALIST       │
│  AGENT POOL       │      │  AGENT POOL       │      │  AGENT POOL       │
├──────────────────┤      ├──────────────────┤      ├──────────────────┤
│ ┌──────────────┐ │      │ ┌──────────────┐ │      │ ┌──────────────┐ │
│ │  Portfolio   │ │      │ │    Risk      │ │      │ │    Tax       │ │
│ │   Advisor    │ │      │ │   Analyst    │ │      │ │   Advisor    │ │
│ └──────────────┘ │      │ └──────────────┘ │      │ └──────────────┘ │
│ ┌──────────────┐ │      │ ┌──────────────┐ │      │ ┌──────────────┐ │
│ │   Budget     │ │      │ │   Market     │ │      │ │  Retirement  │ │
│ │   Planner    │ │      │ │  Researcher  │ │      │ │   Planner    │ │
│ └──────────────┘ │      │ └──────────────┘ │      │ └──────────────┘ │
└──────────────────┘      └──────────────────┘      └──────────────────┘
          │                           │                           │
          └───────────────────────────┼───────────────────────────┘
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                           MCP SERVER LAYER                                  │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐   │
│  │  Market   │ │  Banking  │ │  News &   │ │    Tax    │ │  Document │   │
│  │   Data    │ │   APIs    │ │ Sentiment │ │   Rules   │ │  Analysis │   │
│  │   MCP     │ │    MCP    │ │    MCP    │ │    MCP    │ │    MCP    │   │
│  └───────────┘ └───────────┘ └───────────┘ └───────────┘ └───────────┘   │
└────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                        OBSERVABILITY LAYER                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                      ARIZE PHOENIX                                   │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │  │
│  │  │   Tracing   │  │ Evaluations │  │   Prompt    │  │  Datasets  │  │  │
│  │  │   (OTEL)    │  │  (LLM-as-   │  │ Management  │  │    &       │  │  │
│  │  │             │  │   Judge)    │  │             │  │ Experiments│  │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘  │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                          DATA LAYER                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │  PostgreSQL  │  │    Redis     │  │   Qdrant     │  │     S3       │   │
│  │  (Users,     │  │  (Session,   │  │  (Vector     │  │ (Documents,  │   │
│  │   Profiles)  │  │   Cache)     │  │   Store)     │  │   Backups)   │   │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘   │
└────────────────────────────────────────────────────────────────────────────┘
```

### Agent Architecture Pattern: Supervisor + Specialists

Based on 2025 best practices, we use the **Supervisor Pattern** (not network/swarm):

```
                    ┌─────────────────────┐
                    │   SUPERVISOR AGENT  │
                    │                     │
                    │  • Receives query   │
                    │  • Classifies intent│
                    │  • Delegates tasks  │
                    │  • Aggregates       │
                    │    responses        │
                    │  • Enforces         │
                    │    compliance       │
                    └──────────┬──────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        ▼                      ▼                      ▼
┌───────────────┐      ┌───────────────┐      ┌───────────────┐
│  SPECIALIST   │      │  SPECIALIST   │      │  SPECIALIST   │
│    AGENT      │      │    AGENT      │      │    AGENT      │
├───────────────┤      ├───────────────┤      ├───────────────┤
│ • Single task │      │ • Single task │      │ • Single task │
│ • Own tools   │      │ • Own tools   │      │ • Own tools   │
│ • Own memory  │      │ • Own memory  │      │ • Own memory  │
│ • Reports up  │      │ • Reports up  │      │ • Reports up  │
└───────────────┘      └───────────────┘      └───────────────┘
```

**Why Supervisor Pattern?**
- Deterministic control required for financial compliance
- Clear audit trail for regulatory requirements
- Easier debugging and observability
- Prevents runaway agent loops
- Cost control through centralized orchestration

---

## 1.4 Core Agent Definitions

### Agent Catalog

| Agent | Purpose | Tools (MCP) | Model Tier | Notes |
|-------|---------|-------------|------------|-------|
| **Supervisor** | Intent classification, routing | Internal orchestration | Low (Haiku) | Fast, cheap routing |
| **Fallback Agent** | General queries, help, terminology | System Info, Calculator | Low (Haiku) | Cost-efficient for simple queries |
| **Portfolio Advisor** | Asset allocation, rebalancing recommendations | Market Data, Portfolio MCP | High (Sonnet) | Complex reasoning |
| **Risk Analyst** | Risk assessment, volatility analysis | Market Data, Risk Models MCP | High (Sonnet) | Complex reasoning |
| **Budget Planner** | Income/expense analysis, savings goals | Banking MCP, Calculator | High (Sonnet) | Needs good analysis |
| **Tax Advisor** | Tax optimization, deduction identification | Tax Rules MCP, Calculator | High (Sonnet) | Complex rules |
| **Market Researcher** | News analysis, sentiment, trends | News MCP, Web Search | Medium (Haiku/Sonnet) | Can use Haiku for simple lookups |
| **Retirement Planner** | Long-term projections, Monte Carlo | Calculator, Projections MCP | High (Sonnet) | Complex projections |
| **Compliance Guardian** | Output validation, regulatory checks | Compliance Rules MCP | Low (Haiku) | Rule-based checks |

> **Model Tier Strategy (Validated by MVP):**
> - **Haiku ($0.001/call)**: Classification, routing, simple queries, rule-based checks
> - **Sonnet ($0.01/call)**: Complex analysis, reasoning, financial advice

### Safety Sandwich Architecture

For regulatory compliance, all outputs pass through deterministic validation:

```
┌─────────────────────────────────────────────────────────────┐
│  INPUT LAYER (Deterministic)                                │
│  • Input validation & sanitization                          │
│  • PII detection & masking                                  │
│  • Intent classification                                    │
│  • Rate limiting & abuse detection                          │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  AI LAYER (Probabilistic)                                   │
│  • LLM reasoning                                            │
│  • Agent execution                                          │
│  • Tool calling                                             │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  OUTPUT LAYER (Deterministic)                               │
│  • Compliance validation                                    │
│  • Disclaimer injection                                     │
│  • Hallucination detection                                  │
│  • Audit logging                                            │
│  • Response formatting                                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 1.5 MCP Server Architecture

### MCP vs LangChain Tools (Clarification from MVP)

There are two approaches to providing tools to agents:

| Approach | Implementation | When to Use |
|----------|----------------|-------------|
| **@tool decorator** | LangChain `@tool` in same process | MVP, simple tools, single service |
| **FastMCP Server** | Separate service via MCP protocol | Production, shared tools, microservices |

**MVP Recommendation:** Start with `@tool` decorators for simplicity:
```python
from langchain_core.tools import tool

@tool
def get_stock_quote(symbol: str) -> dict:
    """Get current stock quote."""
    # Direct implementation
    return {"symbol": symbol, "price": 150.00}
```

**Production Migration:** Convert to FastMCP when you need:
- Tools shared across multiple agents/services
- Independent scaling of tool servers
- Different deployment lifecycles

### MCP Best Practices (2025)

Following official MCP specification (v2025-06-18):

1. **Single Responsibility**: Each MCP server has one clear purpose
2. **Stateless Execution**: No session state in servers
3. **OAuth 2.1 Security**: All remote servers use proper auth
4. **OpenTelemetry Integration**: All servers emit traces

### MCP Server Catalog

```python
# MCP Server Registry
mcp_servers = {
    "market-data": {
        "url": "http://mcp-market:8080",
        "tools": ["get_stock_quote", "get_historical_data", "get_market_indices"],
        "auth": "api_key"
    },
    "banking": {
        "url": "http://mcp-banking:8080", 
        "tools": ["get_accounts", "get_transactions", "categorize_spending"],
        "auth": "oauth2"
    },
    "tax-rules": {
        "url": "http://mcp-tax:8080",
        "tools": ["calculate_tax", "get_deductions", "tax_bracket_lookup"],
        "auth": "api_key"
    },
    "news-sentiment": {
        "url": "http://mcp-news:8080",
        "tools": ["search_financial_news", "analyze_sentiment", "get_trending"],
        "auth": "api_key"
    },
    "document-analysis": {
        "url": "http://mcp-docs:8080",
        "tools": ["parse_statement", "extract_holdings", "ocr_document"],
        "auth": "api_key"
    },
    "compliance": {
        "url": "http://mcp-compliance:8080",
        "tools": ["validate_advice", "check_disclaimers", "audit_log"],
        "auth": "internal"
    }
}
```

---

## 1.6 Phoenix Observability Setup

### Tracing Architecture

```python
# Phoenix Configuration (Critical for MVP)
from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor

# Initialize Phoenix tracer
# IMPORTANT: Endpoint must include /v1/traces suffix
tracer_provider = register(
    project_name="finadvisor-ai",
    endpoint="http://localhost:6006/v1/traces",  # Note: /v1/traces is required
)

# Instrument all frameworks (covers LangGraph automatically)
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
```

### Evaluation Framework

```yaml
# Phoenix Evaluators Configuration
evaluators:
  - name: relevance
    type: llm_judge
    model: claude-sonnet-4
    criteria: "Is the financial advice relevant to the user's question?"
    
  - name: factuality
    type: llm_judge
    model: claude-sonnet-4
    criteria: "Are the financial facts and figures accurate?"
    
  - name: compliance
    type: code_based
    function: check_disclaimers
    criteria: "Contains required disclaimers"
    
  - name: safety
    type: llm_judge
    model: claude-sonnet-4
    criteria: "Does not provide specific investment recommendations"
```

### Monitoring Dashboard (Phoenix)

Key metrics to track:
- **Latency**: P50, P95, P99 response times per agent
- **Token usage**: Cost tracking per conversation
- **Error rates**: Tool failures, LLM errors
- **Quality scores**: Evaluation metrics over time
- **Agent paths**: Visualization of agent execution graphs

---

# PHASE 2: MVP Scope Definition

## 2.1 MVP Goals

**Objective:** Prove market fit with minimal viable feature set while maintaining compliance and observability.

### Success Criteria
- [x] User can have a natural conversation about personal finances
- [x] System provides relevant, personalized advice
- [x] All interactions are fully traced in Phoenix
- [ ] Compliance disclaimers are always present
- [x] Response time < 10 seconds for simple queries (actual: 3-5s)
- [x] Cost per conversation < $0.50 (actual: $0.01-0.03)

---

## 2.2 Lessons from Simple MVP (Validated Patterns)

Before diving into features, here are critical lessons learned from building a working MVP:

### What Worked Well

| Pattern | Implementation | Result |
|---------|----------------|--------|
| Supervisor + ReAct | LangGraph StateGraph + create_react_agent | Clean routing, good reasoning |
| Model Tiering | Haiku for routing, Sonnet for analysis | 10x cost reduction vs all-Sonnet |
| Fallback Agent | Full ReAct agent (not simple function) | Much better UX for general queries |
| Singleton Agents | Global agent instances | Faster responses, no recreation |
| Phoenix Tracing | Auto-instrumentation | All spans captured correctly |

### Critical Fixes Discovered

1. **Phoenix endpoint requires `/v1/traces` suffix** - not just base URL
2. **Agent nodes must be `async`** - use `await agent.ainvoke()`
3. **`create_react_agent` uses `prompt` parameter** - not `state_modifier` (API changed)
4. **Response extraction pattern needed** - iterate reversed messages to find content

### Recommended Incremental Approach

```
Week 1-2: Supervisor + 1 Specialist + Fallback Agent (prove architecture)
Week 3-4: Add 2 more specialist agents
Week 5-6: Add remaining specialists + compliance
Week 7-8: Polish, UI, evaluation
```

### Essential MVP Components (Often Overlooked)

1. **Demo UI** - Simple HTML/JS for stakeholder demos (we built one in 1 file)
2. **CLI Tool** - Quick testing without UI (`python cli.py "query"`)
3. **Health Endpoint** - `/health` showing component status
4. **Test Script** - Verify tools work without LLM (`test_flow.py`)

---

## 2.4 MVP Feature Scope

### In Scope (MVP v0.1)

| Feature | Priority | Agent(s) | MCP Server(s) |
|---------|----------|----------|---------------|
| **Budget Analysis** | P0 | Budget Planner | Banking MCP (mock) |
| **Basic Investment Q&A** | P0 | Portfolio Advisor, Market Researcher | Market Data MCP |
| **Savings Goal Planning** | P0 | Budget Planner | Calculator |
| **Risk Assessment Quiz** | P1 | Risk Analyst | Risk Models MCP |
| **Tax Basics** | P1 | Tax Advisor | Tax Rules MCP |
| **Full Observability** | P0 | All | Phoenix |

### Out of Scope (Post-MVP)

- Real bank account integration (Plaid)
- Automated trading/rebalancing
- Complex tax planning
- Estate planning
- Insurance recommendations
- Real-time market alerts
- Mobile native app

---

## 2.5 MVP Architecture

### Simplified Architecture for MVP

```
┌─────────────────────────────────────────────────────────────┐
│                      WEB CLIENT                              │
│              (React + TailwindCSS)                          │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    FASTAPI BACKEND                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │    Auth     │  │  WebSocket  │  │    REST     │         │
│  │  (Simple)   │  │   Handler   │  │    API      │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              SUPERVISOR AGENT (LangGraph)                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Intent → Route → Execute → Validate → Respond      │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                  │
│     ┌─────────────────────┼─────────────────────┐           │
│     ▼                     ▼                     ▼           │
│  ┌──────────┐      ┌──────────┐      ┌──────────┐          │
│  │ Portfolio│      │  Budget  │      │   Risk   │          │
│  │ Advisor  │      │ Planner  │      │ Analyst  │          │
│  └──────────┘      └──────────┘      └──────────┘          │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    MCP SERVERS (MVP)                         │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐               │
│  │  Market   │  │ Calculator│  │ Compliance│               │
│  │   Data    │  │   (math)  │  │  (rules)  │               │
│  │  (mock)   │  │           │  │           │               │
│  └───────────┘  └───────────┘  └───────────┘               │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    ARIZE PHOENIX                             │
│  ┌───────────────────────────────────────────────────────┐ │
│  │  Traces │ Evaluations │ Prompt Mgmt │ Experiments     │ │
│  └───────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      DATA STORES                             │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐   │
│  │   SQLite      │  │    Redis      │  │   Qdrant      │   │
│  │  (User data)  │  │   (Cache)     │  │  (Vectors)    │   │
│  └───────────────┘  └───────────────┘  └───────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### MVP Technology Stack (Simplified)

| Component | MVP Choice | Production Path |
|-----------|------------|-----------------|
| Database | SQLite | → PostgreSQL |
| Cache | Redis (local) | → Redis Cluster |
| Vector Store | Qdrant (local) | → Qdrant Cloud |
| LLM | Claude Sonnet 4 | → Multi-model |
| Hosting | Docker Compose | → Kubernetes |
| Phoenix | Self-hosted (Docker) | → Phoenix Cloud |

---

## 2.4 Cost Estimation (MVP) - Updated from Real Testing

### Development Costs (Estimated)
- **LLM API**: ~$50-100/month during development (lower than expected)
- **Infrastructure**: ~$50/month (small VPS or local)
- **Phoenix**: Free (self-hosted)
- **Data APIs**: Free tiers for MVP (Finnhub, Alpha Vantage, etc.)

### Per-Conversation Cost (Validated by MVP)

| Component | Model | Cost per Call | Notes |
|-----------|-------|---------------|-------|
| Intent Classification | Haiku | ~$0.001 | Fast, cheap |
| Fallback Agent | Haiku | ~$0.002 | Simple queries |
| Specialist Agent | Sonnet | ~$0.01 | Complex analysis |
| **Total (simple query)** | | **~$0.003** | Fallback only |
| **Total (complex query)** | | **~$0.012** | Classifier + Specialist |

### Actual MVP Metrics
- Average tokens: ~1,500 input + 500 output (less than estimated)
- Haiku: ~$0.00025 input + $0.00125 output per 1K tokens
- Sonnet: ~$0.003 input + $0.015 output per 1K tokens
- **Actual: $0.01-0.03 per conversation** (much lower than $0.15 estimate)

---

## 2.5 MVP Approval Checkpoint

### Questions for Stakeholder Approval

1. **Scope Confirmation**: Is the MVP feature set sufficient to demonstrate value?
2. **Timeline**: Target 6-8 weeks for MVP delivery - acceptable?
3. **Budget**: Estimated $500-1000 for MVP development costs - approved?
4. **Technology**: LangGraph + Phoenix + Claude - any constraints?
5. **Compliance**: Mock compliance rules sufficient for MVP, or need legal review?

---

# PHASE 3: User Stories & Implementation

## 3.1 Epic Overview

```
EPIC-001: User Onboarding & Profiling
EPIC-002: Conversational Financial Assistant  
EPIC-003: Budget Analysis & Planning
EPIC-004: Investment Guidance
EPIC-005: Observability & Monitoring
EPIC-006: Compliance & Safety
```

---

## 3.2 User Stories (EARS Format)

### EPIC-001: User Onboarding & Profiling

#### US-001: User Registration
**User Story:** As a new user, I want to create an account so that I can access personalized financial advice.

**Acceptance Criteria (EARS):**
1. WHEN the user accesses the application THEN the system SHALL display a registration form
2. WHEN the user submits valid credentials THEN the system SHALL create an account and redirect to onboarding
3. WHEN the user submits invalid credentials THEN the system SHALL display specific error messages
4. WHEN registration succeeds THEN the system SHALL send a welcome email

**Tasks:**
- [ ] Create user registration API endpoint
- [ ] Implement email validation
- [ ] Set up user database schema
- [ ] Create registration UI component

---

#### US-002: Risk Profile Assessment
**User Story:** As a new user, I want to complete a risk assessment quiz so that the system can personalize advice to my risk tolerance.

**Acceptance Criteria (EARS):**
1. WHEN a new user completes registration THEN the system SHALL prompt for risk assessment
2. WHEN the user answers all questions THEN the system SHALL calculate a risk score (1-10)
3. WHEN the risk score is calculated THEN the system SHALL store it in the user profile
4. IF the user skips assessment THEN the system SHALL assign a default moderate risk profile

**Tasks:**
- [ ] Design risk assessment questionnaire (10 questions)
- [ ] Implement scoring algorithm
- [ ] Create quiz UI with progress indicator
- [ ] Store results in user profile

---

### EPIC-002: Conversational Financial Assistant

#### US-003: Natural Language Query
**User Story:** As a user, I want to ask financial questions in natural language so that I can get advice without learning complex terminology.

**Acceptance Criteria (EARS):**
1. WHEN the user sends a message THEN the system SHALL respond within 10 seconds
2. WHEN the query is financial in nature THEN the system SHALL route to appropriate specialist agent
3. WHEN the query is off-topic THEN the system SHALL politely redirect to financial topics
4. WHEN the system provides advice THEN it SHALL include appropriate disclaimers

**Tasks:**
- [ ] Implement supervisor agent with intent classification
- [ ] Create specialist agent routing logic
- [ ] Set up WebSocket for real-time chat
- [ ] Add response streaming for long answers

---

#### US-004: Conversation Memory
**User Story:** As a user, I want the system to remember context from earlier in our conversation so that I don't have to repeat myself.

**Acceptance Criteria (EARS):**
1. WHEN the user references earlier messages THEN the system SHALL maintain context
2. WHEN a new session starts THEN the system SHALL load previous conversation summary
3. IF the conversation exceeds context limits THEN the system SHALL summarize older messages

**Tasks:**
- [ ] Implement short-term memory (Redis)
- [ ] Create conversation summarization agent
- [ ] Add context window management

---

### EPIC-003: Budget Analysis & Planning

#### US-005: Expense Categorization
**User Story:** As a user, I want to input my expenses so that I can see a categorized breakdown of my spending.

**Acceptance Criteria (EARS):**
1. WHEN the user provides expense data THEN the system SHALL categorize into standard categories
2. WHEN categorization is complete THEN the system SHALL display a visual breakdown
3. WHEN the user disagrees with a category THEN they SHALL be able to recategorize

**Tasks:**
- [ ] Create expense input form (manual entry for MVP)
- [ ] Implement categorization logic (Budget Planner agent)
- [ ] Build spending visualization component
- [ ] Add category override capability

---

#### US-006: Savings Goal Setting
**User Story:** As a user, I want to set savings goals so that I can track my progress toward financial objectives.

**Acceptance Criteria (EARS):**
1. WHEN the user creates a goal THEN the system SHALL require target amount and timeline
2. WHEN a goal is created THEN the system SHALL calculate required monthly savings
3. WHEN the user asks about progress THEN the system SHALL provide current status and projections

**Tasks:**
- [ ] Design goal creation UI
- [ ] Implement savings projection calculator (MCP tool)
- [ ] Create progress tracking dashboard

---

### EPIC-004: Investment Guidance

#### US-007: Portfolio Overview
**User Story:** As a user, I want to describe my current investments so that I can receive portfolio analysis.

**Acceptance Criteria (EARS):**
1. WHEN the user provides portfolio details THEN the system SHALL analyze asset allocation
2. WHEN analysis is complete THEN the system SHALL compare to recommended allocation for risk profile
3. WHEN providing recommendations THEN the system SHALL include educational explanations
4. WHEN providing any investment information THEN the system SHALL include required disclaimers

**Tasks:**
- [ ] Create portfolio input form
- [ ] Implement Portfolio Advisor agent
- [ ] Build asset allocation visualization
- [ ] Add compliance disclaimer injection

---

#### US-008: Market Information
**User Story:** As a user, I want to ask about market conditions so that I can make informed decisions.

**Acceptance Criteria (EARS):**
1. WHEN the user asks about a specific stock/ETF THEN the system SHALL provide current data
2. WHEN providing market data THEN the system SHALL cite sources
3. WHEN the user asks for predictions THEN the system SHALL clarify it cannot predict markets

**Tasks:**
- [ ] Integrate Market Data MCP server
- [ ] Implement Market Researcher agent
- [ ] Add source citation to responses
- [ ] Create prediction disclaimer rules

---

### EPIC-005: Observability & Monitoring

#### US-009: Full Trace Capture
**User Story:** As a developer, I want all agent interactions traced so that I can debug and improve the system.

**Acceptance Criteria (EARS):**
1. WHEN any agent executes THEN Phoenix SHALL capture the full trace
2. WHEN a tool is called THEN the trace SHALL include input/output
3. WHEN an error occurs THEN the trace SHALL include error details
4. WHEN a conversation completes THEN all traces SHALL be viewable in Phoenix UI

**Tasks:**
- [ ] Set up Phoenix Docker container
- [ ] Configure OpenTelemetry instrumentation
- [ ] Instrument all agents and tools
- [ ] Create custom spans for business logic

---

#### US-010: Quality Evaluation
**User Story:** As a product owner, I want automatic quality scoring so that I can monitor advice quality over time.

**Acceptance Criteria (EARS):**
1. WHEN a conversation completes THEN the system SHALL run evaluations
2. WHEN evaluations complete THEN results SHALL be stored in Phoenix
3. WHEN quality drops below threshold THEN the system SHALL alert the team

**Tasks:**
- [ ] Configure Phoenix evaluators
- [ ] Create custom financial advice evaluator
- [ ] Set up alerting for quality degradation
- [ ] Build quality dashboard

---

### EPIC-006: Compliance & Safety

#### US-011: Disclaimer Injection
**User Story:** As a compliance officer, I want all financial advice to include disclaimers so that we meet regulatory requirements.

**Acceptance Criteria (EARS):**
1. WHEN investment advice is provided THEN the system SHALL include "not financial advice" disclaimer
2. WHEN tax information is provided THEN the system SHALL recommend consulting a tax professional
3. WHEN specific securities are mentioned THEN the system SHALL include risk disclosure

**Tasks:**
- [ ] Create compliance rules engine (MCP server)
- [ ] Implement disclaimer templates
- [ ] Add output validation layer
- [ ] Test disclaimer injection in all scenarios

---

#### US-012: Content Safety
**User Story:** As a platform operator, I want to prevent harmful outputs so that users are protected.

**Acceptance Criteria (EARS):**
1. WHEN the system generates advice THEN it SHALL NOT recommend specific securities to buy/sell
2. WHEN the user appears in financial distress THEN the system SHALL provide support resources
3. IF harmful content is detected THEN the system SHALL block and log the incident

**Tasks:**
- [ ] Implement output content filter
- [ ] Create financial distress detection
- [ ] Add support resource database
- [ ] Set up incident logging

---

## 3.3 Implementation Phases

### Phase 1: Foundation (Week 1-2)
- Project scaffolding (FastAPI, LangGraph)
- Phoenix setup and basic tracing
- Database schema and migrations
- User authentication (simple JWT)
- Basic chat UI

### Phase 2: Core Agents (Week 3-4)
- Supervisor agent implementation
- Budget Planner agent
- Portfolio Advisor agent
- Risk Analyst agent
- MCP server scaffolding

### Phase 3: Tools & Integration (Week 5-6)
- Market Data MCP (with mock data)
- Calculator MCP tools
- Compliance MCP server
- Agent-tool integration
- Memory and context management

### Phase 4: Polish & Launch (Week 7-8)
- UI/UX refinement
- Evaluation setup
- Performance optimization
- Documentation
- Demo preparation

---

## 3.4 Technical Implementation Details

### Project Structure (Kiro/Claude Code Ready)

```
finadvisor-ai/
├── .kiro/
│   └── specs/
│       ├── requirements.md      # Generated by Kiro
│       ├── design.md           # Architecture decisions
│       └── tasks.md            # Implementation tasks
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py             # FastAPI app entry point
│   │   ├── routes/
│   │   │   ├── auth.py
│   │   │   ├── chat.py         # /chat endpoint
│   │   │   └── profile.py
│   │   └── middleware/
│   │       ├── auth.py
│   │       └── tracing.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── supervisor.py       # Main orchestrator (LangGraph)
│   │   ├── portfolio.py        # Portfolio Advisor (Sonnet)
│   │   ├── budget.py           # Budget Planner (Sonnet)
│   │   ├── risk.py             # Risk Analyst (Sonnet)
│   │   ├── fallback.py         # Fallback Agent (Haiku) ← NEW
│   │   └── compliance.py       # Compliance Guardian (Haiku)
│   ├── mcp_servers/
│   │   ├── market_data.py      # MVP: @tool decorators
│   │   ├── calculator.py       # Basic math tools
│   │   └── compliance.py       # Rule validation
│   ├── core/
│   │   ├── config.py           # Settings management
│   │   ├── llm.py              # LLM abstraction
│   │   ├── memory.py           # Memory management
│   │   └── tracing.py          # Phoenix setup
│   └── models/
│       ├── user.py
│       ├── conversation.py
│       └── portfolio.py
├── static/                      # ← NEW: Demo UI
│   └── index.html              # Simple HTML/JS demo interface
├── cli.py                       # ← NEW: CLI testing tool
├── test_flow.py                 # ← NEW: Test without LLM
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── docker-compose.dev.yml
├── docs/
│   ├── api.md
│   ├── agents.md
│   ├── SCALING_GUIDE.md        # ← NEW: How to add agents
│   └── deployment.md
├── scripts/
│   ├── setup.sh
│   └── seed_data.py
├── pyproject.toml
├── README.md
└── .env.example
```

### Essential MVP Files (Often Overlooked)

| File | Purpose | Why Critical |
|------|---------|--------------|
| `static/index.html` | Demo UI | Stakeholder demos, quick testing |
| `cli.py` | CLI tool | Fast iteration: `python cli.py "query"` |
| `test_flow.py` | Tool testing | Verify tools work without LLM costs |
| `/health` endpoint | Health check | Production readiness, monitoring |

### Key Implementation Snippets

#### Supervisor Agent (LangGraph)

```python
# src/agents/supervisor.py
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic
from typing import TypedDict, Annotated, Literal
import operator

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    intent: str
    user_profile: dict
    analysis_result: dict | None

# IMPORTANT: Agent nodes must be async
async def portfolio_advisor_node(state: AgentState) -> dict:
    """Execute the portfolio advisor agent."""
    agent = get_portfolio_agent()  # Use singleton pattern
    result = await agent.ainvoke({"messages": state["messages"]})

    # Extract response from agent messages
    analysis_result = None
    for msg in reversed(result.get("messages", [])):
        if hasattr(msg, "content") and msg.content:
            analysis_result = {"response": msg.content}
            break

    return {
        "messages": result.get("messages", []),
        "analysis_result": analysis_result
    }

# Routing function returns node names
def route_to_specialist(state: AgentState) -> Literal["portfolio_advisor", "budget_planner", "fallback_agent"]:
    intent = state.get("intent", "GENERAL")
    routes = {
        "PORTFOLIO": "portfolio_advisor",
        "BUDGET": "budget_planner",
        "RISK": "risk_analyst",
    }
    return routes.get(intent, "fallback_agent")

def create_supervisor_graph():
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("intent_classifier", classify_intent)
    workflow.add_node("portfolio_advisor", portfolio_advisor_node)
    workflow.add_node("budget_planner", budget_planner_node)
    workflow.add_node("risk_analyst", risk_analyst_node)
    workflow.add_node("fallback_agent", fallback_agent_node)  # For general queries
    workflow.add_node("compliance_check", compliance_guardian)
    workflow.add_node("response_formatter", format_response)

    # Define edges
    workflow.add_edge(START, "intent_classifier")

    workflow.add_conditional_edges(
        "intent_classifier",
        route_to_specialist,
        {
            "portfolio_advisor": "portfolio_advisor",
            "budget_planner": "budget_planner",
            "risk_analyst": "risk_analyst",
            "fallback_agent": "fallback_agent"
        }
    )

    # All specialists go through compliance
    for agent in ["portfolio_advisor", "budget_planner", "risk_analyst"]:
        workflow.add_edge(agent, "compliance_check")

    # Fallback doesn't need compliance (no financial advice)
    workflow.add_edge("fallback_agent", "response_formatter")
    workflow.add_edge("compliance_check", "response_formatter")
    workflow.add_edge("response_formatter", END)

    return workflow.compile()

# Singleton pattern - avoid recreating agents per request
_supervisor = None

def get_supervisor():
    global _supervisor
    if _supervisor is None:
        _supervisor = create_supervisor_graph()
    return _supervisor
```

#### Phoenix Tracing Setup

```python
# src/core/tracing.py
from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor

def setup_tracing(project_name: str = "finadvisor-ai"):
    """
    Initialize Phoenix tracing for all components.

    IMPORTANT: Endpoint must include /v1/traces suffix.
    """
    # Register tracer with Phoenix
    tracer_provider = register(
        project_name=project_name,
        endpoint="http://localhost:6006/v1/traces",  # /v1/traces required!
    )

    # Instrument LangChain/LangGraph (covers all agent calls)
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

    print(f"[Tracing] Phoenix initialized - View at: http://localhost:6006")
    return tracer_provider
```

#### Tool Implementation (MVP: @tool decorator)

```python
# src/mcp_servers/market_data.py (MVP approach - simpler)
import os
import requests
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

@tool
def get_stock_quote(symbol: str) -> dict:
    """Get current stock quote for a symbol.

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'NVDA')

    Returns:
        Current price, high, low, and change percent
    """
    symbol = symbol.upper().strip()

    if not FINNHUB_API_KEY:
        # Fallback to mock data if no API key
        return {"symbol": symbol, "price": 150.00, "source": "mock"}

    try:
        response = requests.get(
            "https://finnhub.io/api/v1/quote",
            params={"symbol": symbol, "token": FINNHUB_API_KEY},
            timeout=5
        )
        data = response.json()

        return {
            "symbol": symbol,
            "price": round(data["c"], 2),
            "high": round(data["h"], 2),
            "low": round(data["l"], 2),
            "change_percent": round(((data["c"] - data["pc"]) / data["pc"]) * 100, 2),
            "source": "finnhub"
        }
    except Exception as e:
        return {"error": str(e), "symbol": symbol}
```

#### MCP Server (Production: FastMCP)

```python
# src/mcp_servers/market_data/server.py (Production approach)
from fastmcp import FastMCP

mcp = FastMCP("Market Data Server")

@mcp.tool()
async def get_stock_quote(symbol: str) -> dict:
    """Get current stock quote for a symbol."""
    # Same implementation as above, but as MCP server
    ...

if __name__ == "__main__":
    mcp.run()
```

#### Fallback Agent Pattern (Critical for UX)

Instead of a simple function for "general" queries, use a full ReAct agent:

```python
# src/agents/fallback_agent.py
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

@tool
def get_system_info() -> dict:
    """Get information about system capabilities."""
    return {
        "name": "FinAdvisor AI",
        "capabilities": ["Budget analysis", "Investment Q&A", "Risk assessment"],
        "version": "1.0"
    }

@tool
def get_financial_terminology(term: str) -> dict:
    """Explain common financial terms."""
    glossary = {
        "etf": {"name": "ETF", "definition": "Exchange-Traded Fund..."},
        "roth ira": {"name": "Roth IRA", "definition": "Tax-advantaged retirement account..."},
        # ... more terms
    }
    return glossary.get(term.lower(), {"error": "Term not found"})

FALLBACK_PROMPT = """You are a helpful assistant for FinAdvisor.
Answer general questions, explain terminology, and guide users to the right features.
If asked about specific financial analysis, suggest they ask a specific question."""

def create_fallback_agent():
    # Use Haiku for cost efficiency - simple queries don't need Sonnet
    model = ChatAnthropic(
        model="claude-3-5-haiku-latest",
        max_tokens=1024,
    )
    tools = [get_system_info, get_financial_terminology]
    return create_react_agent(model=model, tools=tools, prompt=FALLBACK_PROMPT)

# Singleton
_fallback_agent = None

def get_fallback_agent():
    global _fallback_agent
    if _fallback_agent is None:
        _fallback_agent = create_fallback_agent()
    return _fallback_agent
```

**Why a full agent instead of a simple responder?**
- Can use tools (terminology lookup, system info)
- Better conversational responses
- Consistent pattern with other agents
- Still cost-efficient with Haiku

---

# PHASE 4: Scalability Guide

## 4.1 Scaling Dimensions

### Horizontal Scaling

```
┌─────────────────────────────────────────────────────────────────┐
│                     LOAD BALANCER (nginx/ALB)                    │
└─────────────────────────────────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌───────────────┐       ┌───────────────┐       ┌───────────────┐
│   API Pod 1   │       │   API Pod 2   │       │   API Pod N   │
│  (Stateless)  │       │  (Stateless)  │       │  (Stateless)  │
└───────────────┘       └───────────────┘       └───────────────┘
        │                       │                       │
        └───────────────────────┼───────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌───────────────┐       ┌───────────────┐       ┌───────────────┐
│  Agent Pod 1  │       │  Agent Pod 2  │       │  Agent Pod N  │
│  (Workers)    │       │  (Workers)    │       │  (Workers)    │
└───────────────┘       └───────────────┘       └───────────────┘
```

### Database Scaling Path

| Stage | Solution | Capacity |
|-------|----------|----------|
| MVP | SQLite | 100 users |
| Growth | PostgreSQL (single) | 10,000 users |
| Scale | PostgreSQL + Read replicas | 100,000 users |
| Enterprise | PostgreSQL cluster + Citus | 1M+ users |

### Vector Store Scaling

| Stage | Solution | Capacity |
|-------|----------|----------|
| MVP | Qdrant (local) | 100K vectors |
| Growth | Qdrant Cloud | 10M vectors |
| Enterprise | Qdrant cluster | 100M+ vectors |

---

## 4.2 Performance Optimization

### Caching Strategy

```python
# Multi-layer caching
CACHE_LAYERS = {
    "l1_memory": {
        "type": "in-memory",
        "ttl": 60,  # seconds
        "use_for": ["frequent_lookups", "session_data"]
    },
    "l2_redis": {
        "type": "redis",
        "ttl": 3600,  # 1 hour
        "use_for": ["user_profiles", "market_data", "llm_responses"]
    },
    "l3_database": {
        "type": "postgresql",
        "ttl": 86400,  # 24 hours
        "use_for": ["historical_data", "audit_logs"]
    }
}
```

### LLM Cost Optimization

1. **Tiered Model Selection**
   - Simple queries → Claude Haiku (cheapest)
   - Complex reasoning → Claude Sonnet
   - Critical compliance → Claude Opus (if needed)

2. **Response Caching**
   - Cache semantic-similar queries
   - Use embedding similarity for cache lookup

3. **Prompt Optimization**
   - Minimize context window usage
   - Use structured output formats

---

## 4.3 Production Deployment

### Kubernetes Manifest (Example)

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: finadvisor-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: finadvisor-api
  template:
    metadata:
      labels:
        app: finadvisor-api
    spec:
      containers:
      - name: api
        image: finadvisor/api:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        env:
        - name: PHOENIX_ENDPOINT
          valueFrom:
            configMapKeyRef:
              name: finadvisor-config
              key: phoenix_endpoint
        - name: ANTHROPIC_API_KEY
          valueFrom:
            secretKeyRef:
              name: finadvisor-secrets
              key: anthropic_api_key
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: finadvisor-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: finadvisor-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

---

## 4.4 Monitoring at Scale

### Key Metrics

```yaml
# Prometheus metrics to expose
metrics:
  # Business metrics
  - conversations_total
  - advice_quality_score
  - user_satisfaction_rating
  
  # Performance metrics  
  - response_latency_seconds
  - tokens_used_total
  - cache_hit_ratio
  
  # Agent metrics
  - agent_execution_duration_seconds
  - tool_calls_total
  - tool_errors_total
  
  # Cost metrics
  - llm_cost_dollars
  - infrastructure_cost_dollars
```

### Alerting Rules

```yaml
# Alert configurations
alerts:
  - name: HighLatency
    condition: response_latency_p95 > 15s
    severity: warning
    
  - name: LowQuality
    condition: advice_quality_score < 0.7
    severity: critical
    
  - name: HighCost
    condition: daily_llm_cost > $100
    severity: warning
    
  - name: AgentLoop
    condition: agent_execution_steps > 10
    severity: critical
```

---

## 4.5 Scaling Checklist

### Pre-Scale Checklist
- [ ] All state externalized (Redis/PostgreSQL)
- [ ] API is stateless and horizontally scalable
- [ ] Database connection pooling configured
- [ ] Caching layer implemented
- [ ] Rate limiting in place
- [ ] Circuit breakers for LLM calls
- [ ] Monitoring and alerting operational

### Scale Triggers
- **10+ concurrent users**: Add API replicas
- **1000+ daily conversations**: Move to PostgreSQL
- **10,000+ daily conversations**: Add read replicas, caching
- **$1000+/month LLM costs**: Implement response caching, model tiering

---

# Appendices

## Appendix A: Technology Alternatives

| Component | Primary Choice | Alternatives |
|-----------|---------------|--------------|
| Orchestration | LangGraph | CrewAI, AutoGen, Strands |
| LLM | Claude Sonnet 4 | GPT-4o, Llama 3, Mistral |
| Vector DB | Qdrant | Pinecone, Weaviate, Chroma |
| Observability | Phoenix | LangSmith, Opik, Helicone |
| API | FastAPI | Flask, Django, Express |

## Appendix B: Compliance Considerations

### Required Disclaimers
1. "This is for educational purposes only and not financial advice"
2. "Consult a qualified financial advisor before making decisions"
3. "Past performance does not guarantee future results"
4. "Investments involve risk of loss"

### Data Privacy
- User financial data encrypted at rest
- PII masked in logs and traces
- GDPR/CCPA compliance ready
- Data retention policies defined

## Appendix C: References

- [Anthropic: Building Effective AI Agents](https://www.anthropic.com/research/building-effective-agents)
- [MCP Specification (v2025-06-18)](https://modelcontextprotocol.io/specification/2025-06-18)
- [Arize Phoenix Documentation](https://arize.com/docs/phoenix)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Kiro IDE Documentation](https://kiro.dev/docs)

## Appendix D: Working MVP Reference

A complete working MVP demonstrating the patterns in this brief is available:

**Repository:** https://github.com/fabiopiazza59-hue/agentic-flow-demo

**Key Files to Study:**
| File | Pattern Demonstrated |
|------|---------------------|
| `orchestrator.py` | LangGraph StateGraph, routing |
| `agents/scalp_agent.py` | ReAct agent with tools |
| `agents/fallback_agent.py` | Fallback agent pattern |
| `core/tracing.py` | Phoenix setup |
| `mcp_servers/market_data.py` | @tool decorator pattern |
| `static/index.html` | Demo UI |
| `docs/SCALING_GUIDE.md` | How to add agents |

**To Run:**
```bash
git clone https://github.com/fabiopiazza59-hue/agentic-flow-demo
cd agentic-flow-demo/agentic-example/simple-MVP
pip install -r requirements.txt
# Add API keys to .env
python -m phoenix.server.main serve  # Terminal 1
python main.py                        # Terminal 2
open http://localhost:8000            # Demo UI
```

---

# Next Steps

**Status (Updated after Simple MVP):**

1. ✅ **Phase 1 (Best Practices & HLD)**: Architecture validated by working MVP
2. ✅ **Phase 2 (MVP Scope)**: Patterns proven with Scalp Trading Assistant
3. ⏳ **Phase 3 (Implementation)**: Ready to implement full FinAdvisor
4. ⏳ **Phase 4 (Scalability)**: Patterns documented in SCALING_GUIDE.md

**What's Been Validated:**
- ✅ LangGraph StateGraph orchestration works
- ✅ ReAct agents with tools work well
- ✅ Model tiering (Haiku/Sonnet) reduces costs 10x
- ✅ Phoenix tracing captures all spans
- ✅ Fallback agent pattern improves UX
- ✅ Demo UI enables stakeholder demos
- ✅ Cost: $0.01-0.03/query (much lower than estimated)

**Ready to Proceed:**
- Reference implementation: https://github.com/fabiopiazza59-hue/agentic-flow-demo
- All patterns documented with working code
- Scaling guide available for adding agents

This document is ready to use with **Kiro** (spec-driven development) or **Claude Code** for full implementation.
