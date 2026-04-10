#!/usr/bin/env python3
"""Thin wrapper for the MCP server entrypoint."""

import asyncio

from entrypoints.mcp_server_main import amain


if __name__ == "__main__":
    asyncio.run(amain())
