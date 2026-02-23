# ─────────────────────────────────────────────────────────────
# Frontend — Multi-stage Dockerfile (Build + Nginx)
# ─────────────────────────────────────────────────────────────

# ── Stage 1: Build ────────────────────────────────────────────
FROM node:20-alpine AS builder

WORKDIR /app

# Install deps first (layer cache)
COPY package.json package-lock.json ./
RUN npm ci

# Copy source & build
COPY . .
RUN npm run build

# ── Stage 2: Serve with Nginx ────────────────────────────────
FROM nginx:1.27-alpine

# Remove default config
RUN rm /etc/nginx/conf.d/default.conf

# Copy our template (envsubst will resolve vars at runtime)
COPY nginx.conf.template /etc/nginx/templates/default.conf.template

# Copy built assets from builder
COPY --from=builder /app/dist /usr/share/nginx/html

# Nginx on alpine uses envsubst automatically from /etc/nginx/templates/
# It reads NGINX_ENVSUBST_TEMPLATE_DIR and outputs to /etc/nginx/conf.d/
# We just need to declare the env vars so envsubst picks them up.
ENV FRONTEND_PORT=9489
ENV BACKEND_HOST=backend
ENV BACKEND_PORT=9867
ENV ENGINE_HOST=ai-engine
ENV ENGINE_PORT=3490

EXPOSE ${FRONTEND_PORT}

# Default CMD — nginx image handles envsubst + start
CMD ["nginx", "-g", "daemon off;"]
