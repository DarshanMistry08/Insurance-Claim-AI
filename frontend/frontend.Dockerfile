# Use the official Node.js image as a parent image
FROM node:20-alpine AS builder

# Set the working directory
WORKDIR /app

# Copy the package.json and package-lock.json (if available)
COPY package.json package-lock.json* ./

# Install dependencies
RUN npm ci

# Copy the rest of the application code
COPY . .

# Build the Next.js application
RUN npm run build

# Production image, copy all the files and run next
FROM node:20-alpine AS runner
WORKDIR /app

ENV NODE_ENV production
# Uncomment the following line in case you want to disable telemetry during runtime.
ENV NEXT_TELEMETRY_DISABLED 1

# Automatically leverage output traces to reduce image size
# https://nextjs.org/docs/advanced-features/output-file-tracing
# COPY --from=builder /app/public ./public
# COPY --from=builder /app/.next/standalone ./
# COPY --from=builder /app/.next/static ./.next/static

# As we don't have standalone output configured by default in next.config.ts, we'll copy everything
COPY --from=builder /app ./

# Expose the listening port
EXPOSE 3000

# Start Next.js
CMD ["npm", "run", "start"]
