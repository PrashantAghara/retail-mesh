from fastapi import FastAPI

app = FastAPI(title="RetailMesh")


@app.get("/health")
def health():
    return {"status": "ok"}
