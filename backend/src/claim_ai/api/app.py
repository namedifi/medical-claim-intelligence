from fastapi import FastAPI

from claim_ai.api.routes.precheck import router as precheck_router
from claim_ai.api.routes.review import create_review_router
from claim_ai.demo.service import ReviewService


def create_app() -> FastAPI:
    app = FastAPI(title="Medical Claim Intelligence", version="0.1.0")
    app.include_router(precheck_router)
    app.include_router(create_review_router(ReviewService.from_default_fixture()))

    @app.get("/health/live")
    def live() -> dict[str, str]:
        return {"status": "ok"}

    return app
