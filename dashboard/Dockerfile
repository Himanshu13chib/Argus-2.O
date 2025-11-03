# Multi-stage build for React Dashboard
FROM node:18-alpine as build

# Set working directory
WORKDIR /app

# Copy package files
COPY dashboard/package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy source code
COPY dashboard/ ./

# Build the application
RUN npm run build

# Production stage
FROM nginx:alpine as production

# Copy built application
COPY --from=build /app/build /usr/share/nginx/html

# Copy nginx configuration
COPY dashboard/nginx.conf /etc/nginx/conf.d/default.conf

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:3000 || exit 1

# Start nginx
CMD ["nginx", "-g", "daemon off;"]

# Development stage
FROM node:18-alpine as development

WORKDIR /app

# Copy package files
COPY dashboard/package*.json ./

# Install all dependencies (including dev)
RUN npm install

# Copy source code
COPY dashboard/ ./

# Expose port
EXPOSE 3000

# Start development server
CMD ["npm", "start"]