# NextCandle API

Cryptocurrency pattern detection API powered by deep learning.

## Tech Stack

- **Monorepo**: Turborepo + pnpm
- **API**: NestJS v11 + Fastify v11 + MikroORM + PostgreSQL
- **API**: NestJS v11 + Fastify v11 + MikroORM + PostgreSQL
- **Testing**: Vitest + Testcontainers

## Project Structure

```
nextcandle-api/
├── apps/
│   └── api/              # NestJS API server
├── turbo.json            # Turborepo config
└── package.json          # Root package
```

## Getting Started

### Prerequisites

- Node.js >= 20
- pnpm >= 9
- Python >= 3.11
- PostgreSQL 16
- CUDA (for GPU support)
- Docker (for Testcontainers)

### Installation

```bash
# Install dependencies
pnpm install

# Install dependencies
pnpm install
```

### Development

```bash
# Start all services
pnpm dev

# Start API only
pnpm api:dev

# Start API only
pnpm api:dev
```

### Database Setup

```bash
# Run migrations
cd apps/api
pnpm migration:up
```

### Testing

```bash
# Run all tests
pnpm test

# API tests
cd apps/api
pnpm test           # All tests
pnpm test:unit      # Unit tests only
pnpm test:integration  # Integration tests (requires Docker)
pnpm test:cov       # With coverage
```

## API Endpoints

### NestJS API (port 4000)

- `GET /` - Health check
- `GET /health` - Detailed health
- `GET /api/docs` - Swagger documentation

- `GET /api/docs` - Swagger documentation

## Infrastructure

Infrastructure is managed in a separate repository: `nextcandle-infra`

## License

Private
