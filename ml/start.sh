#!/bin/bash
cd /opt/render/project/src/ml
exec uvicorn service.main:app --host 0.0.0.0 --port $PORT
