"""MCP (Model Context Protocol) integration package.

Modules:
    mcp_server_registry     Read-only catalog of platform-supported MCP servers
    user_mcp_credential_store CRUD for per-user MCP server credentials
    token_crypto            Fernet-based token encryption for credentials
    user_mcp_pool           Per-user MCP client pool with health tracking
    mcp_skill               Skill plugin that wires MCP tools into the LangGraph agent
"""
