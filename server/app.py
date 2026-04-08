"""OpenEnv SRE — Server entry point for multi-mode deployment."""

import uvicorn


def main():
    """Start the FastAPI server programmatically."""
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=7860,
        workers=1,
    )


if __name__ == "__main__":
    main()
