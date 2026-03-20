# 🌉 LLM Bridge

**AI Gateway Enterprise para o mercado regulado LATAM** — construído sobre LiteLLM, alinhado ao Microsoft Well-Architected Framework.

## O que é

LLM Bridge é um gateway de IA que unifica 100+ provedores de LLM em uma API OpenAI-compatible, adicionando camadas enterprise de segurança, custo, compliance e orquestração.

Inspirado na Bridge do Bradesco, projetado para instituições financeiras e empresas reguladas na América Latina.

## Features por Pilar WAF

### 🔵 Reliability
- **Circuit Breaker** por provider com state machine (CLOSED→OPEN→HALF_OPEN)
- **Chaos Engineering** — 4 tipos de falha simuláveis (provider failure, latency, rate limit, budget)
- **Health probes** avançados (GET /info com memory, uptime, cache stats)
- **Multi-worker** com Gunicorn + worker recycling

### 🔴 Security
- **SSO Corporativo** — OIDC, SAML 2.0, Microsoft Entra ID, MFA enforcement
- **Audit Trail** completo — append-only, filtros, export CSV
- **Compliance LATAM** — PII masking (CPF, CNPJ, phone, cartão), data residency Brasil
- **AI Content Firewall** — 15 regras built-in (prompt injection, jailbreak, data exfiltration, PII leak, toxic)

### 💰 Cost Optimization
- **Cost Router Inteligente** — classifica complexidade, roteia para modelo mais barato
- **Semantic Caching** — cache por similaridade semântica cross-provider (economia 30-50%)
- **FinOps Dashboard** — spend analytics, forecast, budget alerting
- **FinOps Copilot** — recomendações automáticas de otimização

### ⚙️ Operational Excellence
- **Prompt Management** — versionamento, A/B testing, rollback
- **MCP Server Gateway** — registry, health checks, rate limiting, tool discovery
- **Observability Graph** — execution DAG, critical path, anomaly detection
- **IaC** — Helm chart, Dockerfile production, HPA, PDB

### 🚀 Performance
- **Agentic Gateway** — orquestração multi-agente, routing semântico, session tracking
- **Multi-worker** — Gunicorn com worker recycling
- **Autoscaling** — HPA + KEDA configs

## Quick Start

### Docker Compose
```bash
docker compose up -d
```

### Helm (AKS)
```bash
helm install llmbridge deploy/helm/llmbridge/ \
  --set postgresql.auth.password=mypassword \
  --set config.masterKey=sk-my-master-key
```

### API
```bash
curl http://localhost:4000/info

curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-my-key" \
  -d '{"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Hello"}]}'
```

## Arquitetura

```
LiteLLM (100+ providers)
  └── LLM Bridge Extensions
       ├── Security: SSO, Audit, Compliance, Firewall
       ├── Cost: Router, Semantic Cache, FinOps
       ├── OpEx: Prompts, MCP, Observability
       ├── Reliability: Circuit Breaker, Chaos
       └── Performance: Agentic Gateway, Multi-worker
```

## API Endpoints Novos

| Método | Path | Descrição |
|--------|------|-----------|
| GET | /info | System health metrics |
| GET | /audit/logs | Audit trail (paginado, filtros) |
| GET | /audit/logs/export | Export CSV |
| GET | /finops/summary | Spend analytics |
| GET | /finops/trends | Tendências de custo |
| GET | /finops/forecast | Projeção 30 dias |
| POST | /finops/alerts/configure | Alertas de budget |
| POST | /prompts | Criar prompt |
| GET | /prompts | Listar prompts |
| PUT | /prompts/{name} | Atualizar (nova versão) |
| POST | /prompts/{name}/render | Renderizar com variáveis |
| POST | /prompts/{name}/rollback/{v} | Rollback versão |
| GET | /compliance/latam/status | Status compliance |
| GET | /compliance/latam/report | Relatório compliance |
| POST | /agents/invoke | Invocar agente |
| POST | /agents/register | Registrar agente |
| GET | /agents/stats | Estatísticas de agentes |
| POST | /mcp/registry/servers | Registrar MCP server |
| GET | /mcp/registry/tools | Listar tools MCP |

## Tiers

| Tier | Inclui |
|------|--------|
| Community | Gateway 100+ providers, billing tracking, OTEL |
| Pro | Cost Router, Prompt Mgmt, Semantic Cache, FinOps |
| Enterprise | SSO, HA, Audit, IaC, Guardrails, SLA |
| Regulated | Compliance LATAM/BACEN, data residency BR, on-prem |

## Licença

Baseado em LiteLLM (MIT License)
