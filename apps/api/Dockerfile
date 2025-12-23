# ============================================
# Stage 1: Dependencies
# ============================================
FROM node:24-alpine AS deps

WORKDIR /app

# Install pnpm
RUN corepack enable && corepack prepare pnpm@9.15.1 --activate

# Copy workspace config
COPY pnpm-lock.yaml pnpm-workspace.yaml package.json ./
COPY apps/api/package.json ./apps/api/
COPY packages/shared/package.json ./packages/shared/

# Install dependencies only
RUN pnpm install --frozen-lockfile --prod=false

# ============================================
# Stage 2: Builder
# ============================================
FROM node:24-alpine AS builder

WORKDIR /app

# Install pnpm
RUN corepack enable && corepack prepare pnpm@9.15.1 --activate

# Copy dependencies from deps stage
COPY --from=deps /app/node_modules ./node_modules
COPY --from=deps /app/apps/api/node_modules ./apps/api/node_modules
COPY --from=deps /app/packages/shared/node_modules ./packages/shared/node_modules

# Copy source code
COPY pnpm-lock.yaml pnpm-workspace.yaml package.json turbo.json ./
COPY apps/api ./apps/api
COPY packages/shared ./packages/shared

# Build shared package first, then API
RUN pnpm --filter @nextcandle/shared build && \
    pnpm --filter @nextcandle/api build

# Prune dev dependencies
RUN pnpm prune --prod

# ============================================
# Stage 3: Production
# ============================================
FROM node:24-alpine AS runner

WORKDIR /app

# Add non-root user for security
RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 nestjs

# Copy only production artifacts
COPY --from=builder --chown=nestjs:nodejs /app/apps/api/dist ./dist
COPY --from=builder --chown=nestjs:nodejs /app/apps/api/node_modules ./node_modules
COPY --from=builder --chown=nestjs:nodejs /app/apps/api/package.json ./

# Set environment
ENV NODE_ENV=production
ENV PORT=4000

# Switch to non-root user
USER nestjs

EXPOSE 4000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:4000/health || exit 1

CMD ["node", "dist/main.js"]
