#!/bin/sh
# 빈 data 디렉터리로 처음 기동할 때만 실행됨 (/docker-entrypoint-initdb.d)
# root(superuser): postgres (POSTGRES_USER/POSTGRES_PASSWORD), 앱별 role/DB 는 아래에서 생성
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname postgres <<-EOSQL
	CREATE ROLE rag LOGIN PASSWORD 'password';
	CREATE ROLE litellm LOGIN PASSWORD 'password';
	CREATE DATABASE rag OWNER rag;
	CREATE DATABASE litellm OWNER litellm;
EOSQL

# rag role 은 superuser 가 아니므로 확장은 여기서 미리 생성
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname rag <<-EOSQL
	CREATE EXTENSION IF NOT EXISTS vector;
	CREATE EXTENSION IF NOT EXISTS pgcrypto;
EOSQL
