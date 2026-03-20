# 🌉 LLM Bridge

**AI Gateway Enterprise para o mercado regulado LATAM** — construído sobre LiteLLM, alinhado ao Microsoft Well-Architected Framework.

## O que é

LLM Bridge é um gateway de IA que unifica 100+ provedores de LLM em uma API OpenAI-compatible, adicionando camadas enterprise de segurança, custo, compliance e orquestração.

Inspirado na Bridge do Bradesco, projetado para instituições financeiras e empresas reguladas na América Latina.

---

## Instalação

### Pré-requisitos
- Docker e Docker Compose **ou** Kubernetes (AKS recomendado)
- PostgreSQL 14+ (incluído no Helm chart ou via Docker Compose)
- Redis (opcional — necessário para Semantic Cache)
- Chave de API de pelo menos um provedor LLM (Azure OpenAI, OpenAI, Anthropic, etc)

### Opção 1: Docker Compose (desenvolvimento)

```bash
# 1. Clonar o repositório
git clone https://github.com/fsaito-github/litellm.git
cd litellm

# 2. Criar arquivo de configuração
cat > config.yaml << 'EOF'
model_list:
  - model_name: gpt-4o-mini
    litellm_params:
      model: azure/gpt-4o-mini
      api_base: https://SEU-ENDPOINT.openai.azure.com/
      api_key: os.environ/AZURE_API_KEY
      api_version: "2024-08-01-preview"

  - model_name: gpt-4o
    litellm_params:
      model: azure/gpt-4o
      api_base: https://SEU-ENDPOINT.openai.azure.com/
      api_key: os.environ/AZURE_API_KEY
      api_version: "2024-08-01-preview"

litellm_settings:
  drop_params: true
  set_verbose: false

general_settings:
  master_key: sk-llmbridge-dev-key
EOF

# 3. Criar .env com suas credenciais
cat > .env << 'EOF'
AZURE_API_KEY=sua-chave-azure-openai
LLMBRIDGE_ENABLED=true
LLMBRIDGE_FIREWALL_ENABLED=true
LLMBRIDGE_COST_ROUTER_ENABLED=true
LLMBRIDGE_AUDIT_ENABLED=true
EOF

# 4. Subir o stack
docker compose up -d

# 5. Verificar se está rodando
curl http://localhost:4000/info
curl http://localhost:4000/health/liveliness
```

### Opção 2: Docker Production (multi-worker)

```bash
# Build da imagem de produção
docker build -f deploy/Dockerfile.production -t llmbridge:latest .

# Rodar com multi-worker
docker run -d \
  --name llmbridge \
  -p 4000:4000 \
  -e DATABASE_URL="postgresql://user:pass@host:5432/litellm" \
  -e LITELLM_MASTER_KEY="sk-sua-master-key" \
  -e AZURE_API_KEY="sua-chave" \
  -e GUNICORN_WORKERS=4 \
  -e GUNICORN_TIMEOUT=120 \
  -e GUNICORN_MAX_REQUESTS=1000 \
  -e LLMBRIDGE_ENABLED=true \
  -v $(pwd)/config.yaml:/app/config.yaml \
  llmbridge:latest
```

### Opção 3: Helm Chart (AKS / Kubernetes)

```bash
# 1. Criar namespace
kubectl create namespace llmbridge

# 2. Criar secret com credenciais
kubectl create secret generic llmbridge-secrets \
  --namespace llmbridge \
  --from-literal=AZURE_API_KEY=sua-chave \
  --from-literal=LITELLM_MASTER_KEY=sk-sua-master-key

# 3. Criar secret do PostgreSQL
kubectl create secret generic llmbridge-postgresql \
  --namespace llmbridge \
  --from-literal=password=senha-segura-db

# 4. Instalar via Helm
helm install llmbridge deploy/helm/llmbridge/ \
  --namespace llmbridge \
  --set postgresql.auth.existingSecret=llmbridge-postgresql \
  --set existingMasterKeySecret=llmbridge-secrets \
  --set existingMasterKeySecretKey=LITELLM_MASTER_KEY \
  --set env.LLMBRIDGE_ENABLED=true \
  --set envFrom[0].secretRef.name=llmbridge-secrets

# 5. Verificar deploy
kubectl get pods -n llmbridge
kubectl port-forward svc/llmbridge 4000:4000 -n llmbridge

# 6. Testar
curl http://localhost:4000/info
```

### Opção 4: pip install (SDK/desenvolvimento local)

```bash
pip install -e .

# Rodar proxy local
litellm --config config.yaml --port 4000
```

---

## Variáveis de Ambiente

### LLM Bridge Middleware
| Variável | Default | Descrição |
|----------|---------|-----------|
| `LLMBRIDGE_ENABLED` | `false` | Ativa o middleware de integração |
| `LLMBRIDGE_FIREWALL_ENABLED` | `true` | Content Firewall (prompt injection, jailbreak) |
| `LLMBRIDGE_COST_ROUTER_ENABLED` | `true` | Cost Router (roteamento por complexidade) |
| `LLMBRIDGE_CIRCUIT_BREAKER_ENABLED` | `true` | Circuit Breaker por provider |
| `LLMBRIDGE_COMPLIANCE_ENABLED` | `false` | Compliance LATAM (PII masking, data residency) |
| `LLMBRIDGE_AUDIT_ENABLED` | `true` | Audit logging |
| `LLMBRIDGE_SEMANTIC_CACHE_ENABLED` | `false` | Semantic cache (requer Redis) |

### Gunicorn (Produção)
| Variável | Default | Descrição |
|----------|---------|-----------|
| `GUNICORN_WORKERS` | `2*CPU+1` | Número de workers |
| `GUNICORN_TIMEOUT` | `120` | Timeout por request (segundos) |
| `GUNICORN_MAX_REQUESTS` | `1000` | Requests antes de reciclar worker |
| `PORT` | `4000` | Porta de escuta |

### OpenTelemetry / Azure Monitor
| Variável | Default | Descrição |
|----------|---------|-----------|
| `OTEL_EXPORTER` | `console` | Exporter: `console`, `otlp_http`, `azure_monitor` |
| `APPLICATIONINSIGHTS_CONNECTION_STRING` | — | Connection string do App Insights |

---

## Configuração (config.yaml)

```yaml
# Modelos disponíveis
model_list:
  - model_name: gpt-4o-mini          # Tier barato (Cost Router: simple)
    litellm_params:
      model: azure/gpt-4o-mini
      api_base: https://endpoint.openai.azure.com/
      api_key: os.environ/AZURE_API_KEY

  - model_name: gpt-4o               # Tier premium (Cost Router: complex)
    litellm_params:
      model: azure/gpt-4o
      api_base: https://endpoint.openai.azure.com/
      api_key: os.environ/AZURE_API_KEY

# Settings globais
litellm_settings:
  drop_params: true
  callbacks: ["otel"]                 # Ativar OpenTelemetry

general_settings:
  master_key: sk-sua-master-key
  database_url: os.environ/DATABASE_URL
```

---

## Uso da API

### Chat Completion (OpenAI-compatible)
```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-sua-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Olá, tudo bem?"}]
  }'
```

### Health Check
```bash
curl http://localhost:4000/info
# Retorna: version, uptime, memory_rss_mb, active_models_count, cache_stats
```

### FinOps — Consultar gastos
```bash
# Resumo de gastos por org/team/project
curl http://localhost:4000/finops/summary?period=last_30d \
  -H "Authorization: Bearer sk-master-key"

# Projeção de gastos
curl http://localhost:4000/finops/forecast \
  -H "Authorization: Bearer sk-master-key"
```

### Prompt Management
```bash
# Criar prompt
curl -X POST http://localhost:4000/prompts \
  -H "Authorization: Bearer sk-master-key" \
  -H "Content-Type: application/json" \
  -d '{"prompt_name": "resumo", "template": "Resuma em {{idioma}}: {{texto}}"}'

# Renderizar prompt com variáveis
curl -X POST http://localhost:4000/prompts/resumo/render \
  -H "Authorization: Bearer sk-master-key" \
  -d '{"variables": {"idioma": "português", "texto": "Long text here..."}}'
```

### Audit Logs
```bash
# Consultar logs de auditoria
curl "http://localhost:4000/audit/logs?action=llm.completion&limit=10" \
  -H "Authorization: Bearer sk-master-key"

# Exportar CSV
curl "http://localhost:4000/audit/logs/export?start_date=2026-01-01" \
  -H "Authorization: Bearer sk-master-key" > audit.csv
```

### Compliance LATAM
```bash
# Status do compliance
curl http://localhost:4000/compliance/latam/status \
  -H "Authorization: Bearer sk-master-key"

# Relatório (PII detections, residency violations)
curl http://localhost:4000/compliance/latam/report?period=last_30d \
  -H "Authorization: Bearer sk-master-key"
```

### Agent Gateway
```bash
# Registrar agente
curl -X POST http://localhost:4000/agents/register \
  -H "Authorization: Bearer sk-master-key" \
  -d '{"agent_id": "code-assistant", "name": "Code Assistant", "description": "Ajuda com programação", "model": "gpt-4o", "system_prompt": "Você é um assistente de programação expert."}'

# Invocar agente
curl -X POST http://localhost:4000/agents/invoke \
  -H "Authorization: Bearer sk-master-key" \
  -d '{"session_id": "sess-123", "messages": [{"role": "user", "content": "Como fazer um API em FastAPI?"}]}'
```

---

## Arquitetura

```
┌──────────────────────────────────────────────────────────────┐
│                        LLM Bridge                            │
│                                                              │
│  Request ──► Content Firewall ──► PII Mask (input)           │
│          ──► Cost Router ──► Circuit Breaker                 │
│          ──► Semantic Cache lookup                            │
│          ──► LiteLLM (100+ providers)                        │
│          ──► PII Mask (output) ──► Audit Log                 │
│          ──► Semantic Cache store ──► Response                │
│                                                              │
│  ┌─────────────┐ ┌──────────────┐ ┌────────────────┐        │
│  │  Security    │ │    Cost      │ │     OpEx       │        │
│  │ SSO/OIDC    │ │ Cost Router  │ │ Prompt Mgmt    │        │
│  │ Audit Trail │ │ Sem. Cache   │ │ MCP Gateway    │        │
│  │ Compliance  │ │ FinOps       │ │ Obs Graph      │        │
│  │ Firewall    │ │ FinOps AI    │ │ Helm/IaC       │        │
│  └─────────────┘ └──────────────┘ └────────────────┘        │
│  ┌─────────────┐ ┌──────────────┐                            │
│  │ Reliability  │ │ Performance  │                            │
│  │ Cir. Breaker│ │ Agentic GW   │                            │
│  │ Chaos Eng   │ │ Multi-worker │                            │
│  └─────────────┘ └──────────────┘                            │
└──────────────────────────────────────────────────────────────┘
```

## API Endpoints

| Método | Path | Auth | Descrição |
|--------|------|------|-----------|
| GET | `/info` | Não | System health (memory, uptime, cache) |
| GET | `/health/liveliness` | Não | K8s liveness probe |
| GET | `/health/readiness` | Não | K8s readiness probe |
| GET | `/audit/logs` | Admin | Audit trail (paginado, filtros) |
| GET | `/audit/logs/export` | Admin | Export CSV |
| GET | `/finops/summary` | Admin | Spend por org/team/project/model |
| GET | `/finops/trends` | Admin | Spend diário (time series) |
| GET | `/finops/forecast` | Admin | Projeção 30 dias |
| POST | `/finops/alerts/configure` | Admin | Budget alert thresholds |
| POST | `/prompts` | Admin | Criar prompt |
| GET | `/prompts` | Admin | Listar prompts |
| GET | `/prompts/{name}` | Admin | Obter prompt |
| PUT | `/prompts/{name}` | Admin | Atualizar (nova versão) |
| GET | `/prompts/{name}/versions` | Admin | Histórico de versões |
| POST | `/prompts/{name}/render` | Admin | Renderizar com variáveis |
| POST | `/prompts/{name}/rollback/{v}` | Admin | Rollback para versão |
| GET | `/compliance/latam/status` | Admin | Status compliance |
| GET | `/compliance/latam/report` | Admin | Relatório compliance |
| POST | `/agents/register` | Admin | Registrar agente |
| POST | `/agents/invoke` | Admin | Invocar agente |
| GET | `/agents` | Admin | Listar agentes |
| GET | `/agents/stats` | Admin | Estatísticas de agentes |
| GET | `/agents/sessions/{id}` | Admin | Histórico da sessão |
| POST | `/mcp/registry/servers` | Admin | Registrar MCP server |
| GET | `/mcp/registry/servers` | Admin | Listar MCP servers |
| DELETE | `/mcp/registry/servers/{id}` | Admin | Remover MCP server |
| GET | `/mcp/registry/tools` | Admin | Listar tools disponíveis |
| GET | `/mcp/registry/stats` | Admin | Estatísticas MCP |

## Dashboard UI

O dashboard web está disponível em `http://localhost:4000/ui` com 6 páginas adicionais do LLM Bridge:

- **FinOps** — Gráficos de custo por org/team/model, forecast, alertas
- **Audit Logs** — Timeline de ações, filtros, export CSV
- **Compliance** — PII detections, data residency status
- **Agent Gateway** — Lista de agentes, sessões, invocações
- **Content Firewall** — 15 regras, violações, configuração
- **Prompt Management** — Editor, versionamento, render test, rollback

## Testes

```bash
# Rodar todos os testes
pip install pytest pytest-cov
pytest tests/llmbridge/ -v

# Com coverage
pytest tests/llmbridge/ --cov=litellm/proxy/hooks --cov-report=term-missing
```

183 testes cobrindo: circuit breaker, cost router, content firewall, compliance LATAM (CPF/CNPJ), semantic cache, chaos engineering, observability graph, MCP registry, agentic gateway, finops copilot, info endpoint.

## Tiers

| Tier | Inclui |
|------|--------|
| **Community** | Gateway 100+ providers, billing tracking, OTEL básico |
| **Pro** | Cost Router, Prompt Mgmt, Semantic Cache, FinOps Dashboard |
| **Enterprise** | SSO, HA Multi-region, Audit, IaC, Guardrails, SLA |
| **Regulated** | Compliance LATAM/BACEN, data residency BR, on-prem, suporte |

## Licença

Baseado em LiteLLM (MIT License)
