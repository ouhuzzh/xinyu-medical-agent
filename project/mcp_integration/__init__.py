"""MCP (Model Context Protocol) integration package.

Modules:
    token_crypto       Fernet-based token encryption for hospital credentials
    hospital_registry  Read-only catalog of platform-supported hospitals
    user_hospital_store CRUD for per-user hospital credentials
    user_mcp_pool      Per-user MCP client pool with health tracking
    mcp_skill          Skill plugin that wires MCP tools into the LangGraph agent
"""
