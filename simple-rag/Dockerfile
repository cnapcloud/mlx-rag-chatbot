FROM node:22-alpine

WORKDIR /app

# 의존성 레이어 캐시: package*.json 이 바뀔 때만 재설치 (tsx 가 dependencies 에 있어 런타임에 사용)
COPY package.json package-lock.json ./
RUN npm ci --omit=dev

COPY tsconfig.json ./
COPY src ./src

ENV NODE_ENV=production
EXPOSE 3000

# 테이블이 없으면 생성(idempotent) 후 서버 기동
CMD ["sh", "-c", "npm run migrate && npm start"]
