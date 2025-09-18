from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routers import predict, admin

app = FastAPI(title="Burnout Predictor API")

# CORS for frontend (Next.js runs at localhost:3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(predict.router)
app.include_router(admin.router)
