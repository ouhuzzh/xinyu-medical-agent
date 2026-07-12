"""Process entry point for background maintenance workers."""

from core.knowledge_base_worker import run_worker


if __name__ == "__main__":
    run_worker()

