import logging
import shutil
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status

import config
from api.auth import (
    AuthenticatedUser,
    enforce_sync_rate_limit,
    enforce_upload_rate_limit,
    require_admin_user,
)
from api.dependencies import get_container
from api.schemas import (
    DocumentItem,
    DocumentListResponse,
    DocumentSourceCoverageResponse,
    DocumentStatusResponse,
    DocumentTaskItem,
    DocumentTaskListResponse,
    DocumentUploadResponse,
    KnowledgeBaseStatusResponse,
    OfficialSyncRequest,
    OfficialSyncResponse,
)
from core.document_parsers import supported_upload_extensions


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/documents", tags=["documents"])


def _knowledge_response(container) -> KnowledgeBaseStatusResponse:
    knowledge = container.rag_system.get_knowledge_base_status()
    return KnowledgeBaseStatusResponse(
        status=knowledge["status"],
        message=knowledge["message"],
        last_error=knowledge.get("last_error") or "",
        stats=knowledge.get("stats") or {},
    )


def _recent_tasks(container) -> list[dict]:
    stats = container.rag_system.get_knowledge_base_status().get("stats") or {}
    return list(stats.get("recent_imports") or [])


def _task_item_from_event(event: dict) -> DocumentTaskItem:
    return DocumentTaskItem(
        source=str(event.get("source") or ""),
        label=str(event.get("label") or event.get("source") or "同步任务"),
        status=str(event.get("status") or "completed"),
        timestamp=str(event.get("timestamp") or ""),
        downloaded=int(event.get("downloaded") or 0),
        written=int(event.get("written") or 0),
        updated=int(event.get("updated") or 0),
        deactivated=int(event.get("deactivated") or 0),
        unchanged=int(event.get("unchanged") or 0),
        skipped=int(event.get("skipped") or 0),
        failed=int(event.get("failed") or 0),
        index_added=int(event.get("index_added") or 0),
        index_skipped=int(event.get("index_skipped") or 0),
        duration_ms=float(event.get("duration_ms") or 0),
        note=str(event.get("note") or ""),
        trigger_type=str(event.get("trigger_type") or "manual"),
        scope=str(event.get("scope") or ""),
        conversion_details=[str(item) for item in (event.get("conversion_details") or [])],
        failure_details=[str(item) for item in (event.get("failure_details") or [])],
    )


def _task_items(container) -> list[DocumentTaskItem]:
    return [_task_item_from_event(event) for event in _recent_tasks(container)]


def _source_coverage(container) -> list[dict]:
    getter = getattr(container.document_manager, "get_official_source_coverage", None)
    if not callable(getter):
        return []
    try:
        return list(getter())
    except Exception:
        logger.warning("Failed to read official source coverage", exc_info=True)
        return []


def _safe_upload_name(filename: str) -> str:
    name = Path(filename or "").name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="文件名不能为空")
    suffix = Path(name).suffix.lower()
    if suffix not in supported_upload_extensions():
        raise HTTPException(status_code=400, detail=f"不支持的文件类型：{suffix or 'unknown'}")
    return name


def _document_item_from_inventory(item: dict) -> DocumentItem:
    modified_at = item.get("modified_at") or ""
    if isinstance(modified_at, (int, float)):
        modified_at = datetime.fromtimestamp(modified_at).isoformat(timespec="seconds")
    return DocumentItem(
        name=str(item.get("name") or ""),
        file_type=str(item.get("file_type") or "md"),
        size_bytes=int(item.get("size_bytes") or 0),
        modified_at=str(modified_at),
        title=str(item.get("title") or ""),
        source_name=str(item.get("source_name") or ""),
        source_type=str(item.get("source_type") or ""),
        source_key=str(item.get("source_key") or ""),
        sync_status=str(item.get("sync_status") or ""),
        is_active=bool(item.get("is_active", True)),
        freshness_bucket=str(item.get("freshness_bucket") or ""),
        original_url=str(item.get("original_url") or ""),
    )


def _set_request_context(request: Request, route_type: str):
    request.state.route_type = route_type


@router.get("/status", response_model=DocumentStatusResponse)
def documents_status(
    request: Request,
    current_user: AuthenticatedUser = Depends(require_admin_user),
):
    del current_user
    _set_request_context(request, "documents_status")
    container = get_container()
    return DocumentStatusResponse(
        knowledge_base=_knowledge_response(container),
        recent_tasks=_task_items(container),
        source_coverage=_source_coverage(container),
    )


@router.get("/list", response_model=DocumentListResponse)
def documents_list(
    request: Request,
    current_user: AuthenticatedUser = Depends(require_admin_user),
):
    del current_user
    _set_request_context(request, "documents_list")
    container = get_container()
    document_manager = container.document_manager
    inventory_getter = getattr(document_manager, "get_document_inventory", None)
    if callable(inventory_getter):
        return DocumentListResponse(
            documents=[_document_item_from_inventory(item) for item in inventory_getter()]
        )

    items = []
    for path in document_manager.get_markdown_paths():
        stat = path.stat()
        items.append(
            DocumentItem(
                name=path.name,
                file_type=path.suffix.lstrip(".") or "md",
                size_bytes=stat.st_size,
                modified_at=datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
            )
        )
    return DocumentListResponse(documents=items)


@router.get("/tasks", response_model=DocumentTaskListResponse)
def documents_tasks(
    request: Request,
    current_user: AuthenticatedUser = Depends(require_admin_user),
):
    del current_user
    _set_request_context(request, "documents_tasks")
    container = get_container()
    return DocumentTaskListResponse(tasks=_task_items(container))


@router.get("/sources", response_model=DocumentSourceCoverageResponse)
def documents_sources(
    request: Request,
    current_user: AuthenticatedUser = Depends(require_admin_user),
):
    del current_user
    _set_request_context(request, "documents_sources")
    container = get_container()
    return DocumentSourceCoverageResponse(sources=_source_coverage(container))


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_documents(
    request: Request,
    files: list[UploadFile] = File(...),
    current_user: AuthenticatedUser = Depends(require_admin_user),
):
    _set_request_context(request, "documents_upload")
    request.state.user_id = current_user.user_id
    if not files:
        raise HTTPException(status_code=400, detail="请选择要上传的文件")
    if len(files) > config.API_UPLOAD_MAX_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"单次最多上传 {config.API_UPLOAD_MAX_FILES} 个文件。",
        )
    enforce_upload_rate_limit(current_user)

    container = get_container()
    upload_dir = Path("runtime") / "api_uploads" / uuid.uuid4().hex
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    max_size_bytes = config.API_UPLOAD_MAX_FILE_SIZE_MB * 1024 * 1024
    try:
        for file in files:
            filename = _safe_upload_name(file.filename or "")
            target = upload_dir / filename
            total_bytes = 0
            with target.open("wb") as buffer:
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    total_bytes += len(chunk)
                    if total_bytes > max_size_bytes:
                        raise HTTPException(
                            status_code=400,
                            detail=f"{filename} 超过单文件大小限制（{config.API_UPLOAD_MAX_FILE_SIZE_MB} MB）。",
                        )
                    buffer.write(chunk)
            saved_paths.append(target)

        report = container.document_manager.add_documents_with_report(saved_paths)
        sync_event = report.get("sync_event")
        if sync_event:
            container.rag_system.record_import_event(sync_event)
        container.rag_system.refresh_knowledge_base_status()
        return DocumentUploadResponse(
            message=(
                f"已处理 {report.get('processed', 0)} 个文件：新增 {report.get('added', 0)}，"
                f"更新 {report.get('updated', 0)}，未变化 {report.get('unchanged', 0)}，"
                f"失败 {report.get('failed', 0)}。"
            ),
            report=report,
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Document upload failed for user_id=%s", current_user.user_id)
        raise HTTPException(status_code=500, detail="文档上传处理失败，请查看服务日志。")
    finally:
        shutil.rmtree(upload_dir, ignore_errors=True)


@router.post("/sync-official", response_model=OfficialSyncResponse)
def sync_official_documents(
    request: Request,
    payload: OfficialSyncRequest,
    current_user: AuthenticatedUser = Depends(require_admin_user),
):
    _set_request_context(request, "documents_sync_official")
    request.state.user_id = current_user.user_id
    enforce_sync_rate_limit(current_user)
    container = get_container()
    try:
        result = container.document_manager.sync_official_source(
            source=payload.source,
            limit=payload.limit,
            trigger_type="manual",
        )
        if result.status == "skipped_locked":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="已有知识库同步任务正在执行，请稍后再试。",
            )
        event = result.to_event()
        container.rag_system.record_import_event(event)
        container.rag_system.refresh_knowledge_base_status()
        return OfficialSyncResponse(
            message=(
                f"官方同步完成：新增 {result.written}，更新 {result.updated}，"
                f"下线 {result.deactivated}，未变化 {result.unchanged}。"
            ),
            result=event,
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception(
            "Official document sync failed for source=%s user_id=%s",
            payload.source,
            current_user.user_id,
        )
        raise HTTPException(status_code=500, detail="官方资料同步失败，请查看服务日志。")
