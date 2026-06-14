FROM node:24-alpine AS build

WORKDIR /app

COPY frontend/package*.json ./
RUN npm ci

COPY frontend ./

ARG VITE_API_BASE_URL=http://localhost:8000
ARG VITE_API_AUTH_TOKEN=demo-admin-token
ARG VITE_APP_ENV=development
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL
ENV VITE_API_AUTH_TOKEN=$VITE_API_AUTH_TOKEN
ENV VITE_APP_ENV=$VITE_APP_ENV

RUN npm run build

FROM nginx:1.27-alpine

COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80
