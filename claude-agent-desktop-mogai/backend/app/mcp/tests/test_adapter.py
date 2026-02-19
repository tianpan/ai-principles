# -*- coding: utf-8 -*-
"""
Unit tests for MCP Tool Adapter module

Tests cover:
- MCPToolDefinition dataclass
- MCPToolAdapter static methods
- is_mcp_tool()
- to_claude_tool()
- parse_tool_name()
- make_mcp_name()
- create_tool_definition()
"""

import pytest
from typing import Dict, Any

from app.mcp.adapter import MCPToolAdapter, MCPToolDefinition


class TestMCPToolDefinition:
    """Tests for MCPToolDefinition dataclass"""

    def test_create_tool_definition_basic(self):
        """Test basic tool definition creation"""
        tool = MCPToolDefinition(
            name="test_tool",
            mcp_name="mcp__server__test_tool",
            description="A test tool",
            input_schema={"type": "object"},
            server_name="server",
        )

        assert tool.name == "test_tool"
        assert tool.mcp_name == "mcp__server__test_tool"
        assert tool.description == "A test tool"
        assert tool.input_schema == {"type": "object"}
        assert tool.server_name == "server"
        assert tool.annotations == {}

    def test_create_tool_definition_with_annotations(self):
        """Test tool definition with annotations"""
        annotations = {"category": "query", "version": "1.0"}
        tool = MCPToolDefinition(
            name="test_tool",
            mcp_name="mcp__server__test_tool",
            description="A test tool",
            input_schema={"type": "object"},
            server_name="server",
            annotations=annotations,
        )

        assert tool.annotations == annotations

    def test_tool_definition_to_dict(self):
        """Test to_dict() method"""
        tool = MCPToolDefinition(
            name="test_tool",
            mcp_name="mcp__server__test_tool",
            description="A test tool",
            input_schema={"type": "object"},
            server_name="server",
            annotations={"key": "value"},
        )

        result = tool.to_dict()

        assert isinstance(result, dict)
        assert result["name"] == "test_tool"
        assert result["mcp_name"] == "mcp__server__test_tool"
        assert result["description"] == "A test tool"
        assert result["input_schema"] == {"type": "object"}
        assert result["server_name"] == "server"
        assert result["annotations"] == {"key": "value"}


class TestIsMcpTool:
    """Tests for MCPToolAdapter.is_mcp_tool()"""

    def test_is_mcp_tool_valid(self):
        """Test valid MCP tool names"""
        valid_names = [
            "mcp__top__query_station",
            "mcp__knowledge__search_faq",
            "mcp__server__tool",
            "mcp__a__b",
        ]

        for name in valid_names:
            assert MCPToolAdapter.is_mcp_tool(name) is True, f"Failed for: {name}"

    def test_is_mcp_tool_invalid(self):
        """Test invalid MCP tool names"""
        invalid_names = [
            "query_station",
            "top__query_station",
            "mcp_query_station",
            "mcp_top_query_station",
            "",
            "MCP__top__query_station",  # Case sensitive
            "mcp_",
        ]

        for name in invalid_names:
            assert MCPToolAdapter.is_mcp_tool(name) is False, f"Failed for: {name}"

    def test_is_mcp_tool_edge_cases(self):
        """Test edge cases for is_mcp_tool - matches prefix only"""
        # "mcp__" alone technically matches the prefix check
        # The full validation happens in parse_tool_name
        assert MCPToolAdapter.is_mcp_tool("mcp__") is True
        # But parsing should fail
        with pytest.raises(ValueError):
            MCPToolAdapter.parse_tool_name("mcp__")

    def test_is_mcp_tool_empty_string(self):
        """Test empty string"""
        assert MCPToolAdapter.is_mcp_tool("") is False

    def test_is_mcp_tool_none_raises_error(self):
        """Test None raises TypeError"""
        with pytest.raises((TypeError, AttributeError)):
            MCPToolAdapter.is_mcp_tool(None)


class TestToClaudeTool:
    """Tests for MCPToolAdapter.to_claude_tool()"""

    def test_to_claude_tool_basic(self, sample_tool_definition: MCPToolDefinition):
        """Test basic conversion to Claude tool format"""
        result = MCPToolAdapter.to_claude_tool(sample_tool_definition)

        assert isinstance(result, dict)
        assert result["name"] == sample_tool_definition.mcp_name
        assert result["description"] == sample_tool_definition.description
        assert result["input_schema"] == sample_tool_definition.input_schema

    def test_to_claude_tool_with_annotations(self):
        """Test conversion includes annotations when present"""
        tool = MCPToolDefinition(
            name="test_tool",
            mcp_name="mcp__server__test_tool",
            description="A test tool",
            input_schema={"type": "object"},
            server_name="server",
            annotations={"category": "query", "version": "1.0"},
        )

        result = MCPToolAdapter.to_claude_tool(tool)

        assert "annotations" in result
        assert result["annotations"] == {"category": "query", "version": "1.0"}

    def test_to_claude_tool_without_annotations(self):
        """Test conversion without annotations"""
        tool = MCPToolDefinition(
            name="test_tool",
            mcp_name="mcp__server__test_tool",
            description="A test tool",
            input_schema={"type": "object"},
            server_name="server",
            annotations={},
        )

        result = MCPToolAdapter.to_claude_tool(tool)

        # Annotations should not be in result if empty
        assert "annotations" not in result

    def test_to_claude_tool_preserves_input_schema(self):
        """Test that input_schema is preserved correctly"""
        complex_schema = {
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "First parameter"},
                "param2": {"type": "number", "minimum": 0},
                "param3": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["param1"],
        }

        tool = MCPToolDefinition(
            name="test_tool",
            mcp_name="mcp__server__test_tool",
            description="A test tool",
            input_schema=complex_schema,
            server_name="server",
        )

        result = MCPToolAdapter.to_claude_tool(tool)

        assert result["input_schema"] == complex_schema


class TestParseToolName:
    """Tests for MCPToolAdapter.parse_tool_name()"""

    def test_parse_tool_name_valid(self):
        """Test valid tool name parsing"""
        test_cases = [
            ("mcp__top__query_station", ("top", "query_station")),
            ("mcp__knowledge__search_faq", ("knowledge", "search_faq")),
            ("mcp__server__tool", ("server", "tool")),
        ]

        for mcp_name, expected in test_cases:
            server_name, original_name = MCPToolAdapter.parse_tool_name(mcp_name)
            assert (server_name, original_name) == expected, f"Failed for: {mcp_name}"

    def test_parse_tool_name_with_underscores(self):
        """Test parsing tool names with underscores"""
        # Tool name itself contains "__"
        server_name, original_name = MCPToolAdapter.parse_tool_name(
            "mcp__server__my__special__tool"
        )

        assert server_name == "server"
        assert original_name == "my__special__tool"

    def test_parse_tool_name_invalid_format(self):
        """Test invalid format raises ValueError"""
        invalid_names = [
            "query_station",
            "top__query_station",
            "mcp_query_station",
            "mcp__server",  # Missing tool name
            "mcp__",  # Missing server and tool name
        ]

        for name in invalid_names:
            with pytest.raises(ValueError, match="Invalid MCP tool name format|Not an MCP tool"):
                MCPToolAdapter.parse_tool_name(name)

    def test_parse_tool_name_not_mcp_tool(self):
        """Test non-MCP tool name raises ValueError"""
        with pytest.raises(ValueError, match="Not an MCP tool"):
            MCPToolAdapter.parse_tool_name("regular_tool_name")

    def test_parse_tool_name_empty_raises_error(self):
        """Test empty string raises ValueError"""
        with pytest.raises(ValueError):
            MCPToolAdapter.parse_tool_name("")


class TestMakeMcpName:
    """Tests for MCPToolAdapter.make_mcp_name()"""

    def test_make_mcp_name_basic(self):
        """Test basic MCP name creation"""
        result = MCPToolAdapter.make_mcp_name("top", "query_station")

        assert result == "mcp__top__query_station"

    def test_make_mcp_name_various_servers(self):
        """Test MCP name creation with various server names"""
        test_cases = [
            ("top", "query_station", "mcp__top__query_station"),
            ("knowledge", "search_faq", "mcp__knowledge__search_faq"),
            ("my_server", "my_tool", "mcp__my_server__my_tool"),
        ]

        for server, tool, expected in test_cases:
            result = MCPToolAdapter.make_mcp_name(server, tool)
            assert result == expected

    def test_make_mcp_name_with_underscores(self):
        """Test MCP name with underscores in tool name"""
        result = MCPToolAdapter.make_mcp_name("server", "my_special_tool")

        assert result == "mcp__server__my_special_tool"

    def test_make_mcp_name_round_trip(self):
        """Test that make_mcp_name and parse_tool_name are inverses"""
        server_name = "test_server"
        tool_name = "test_tool"

        mcp_name = MCPToolAdapter.make_mcp_name(server_name, tool_name)
        parsed_server, parsed_tool = MCPToolAdapter.parse_tool_name(mcp_name)

        assert parsed_server == server_name
        assert parsed_tool == tool_name


class TestCreateToolDefinition:
    """Tests for MCPToolAdapter.create_tool_definition()"""

    def test_create_tool_definition_basic(self, sample_input_schema: Dict[str, Any]):
        """Test basic tool definition creation"""
        tool = MCPToolAdapter.create_tool_definition(
            server_name="top",
            tool_name="query_station",
            description="Query station information",
            input_schema=sample_input_schema,
        )

        assert isinstance(tool, MCPToolDefinition)
        assert tool.name == "query_station"
        assert tool.mcp_name == "mcp__top__query_station"
        assert tool.description == "Query station information"
        assert tool.input_schema == sample_input_schema
        assert tool.server_name == "top"
        assert tool.annotations == {}

    def test_create_tool_definition_with_annotations(self, sample_input_schema: Dict[str, Any]):
        """Test tool definition creation with annotations"""
        annotations = {"category": "query", "version": "2.0"}

        tool = MCPToolAdapter.create_tool_definition(
            server_name="top",
            tool_name="query_station",
            description="Query station information",
            input_schema=sample_input_schema,
            annotations=annotations,
        )

        assert tool.annotations == annotations

    def test_create_tool_definition_annotations_none(self, sample_input_schema: Dict[str, Any]):
        """Test tool definition creation with None annotations"""
        tool = MCPToolAdapter.create_tool_definition(
            server_name="top",
            tool_name="query_station",
            description="Query station information",
            input_schema=sample_input_schema,
            annotations=None,
        )

        assert tool.annotations == {}

    def test_create_tool_definition_mcp_name_format(self):
        """Test that created tool has correct MCP name format"""
        tool = MCPToolAdapter.create_tool_definition(
            server_name="my_server",
            tool_name="my_tool",
            description="Test tool",
            input_schema={"type": "object"},
        )

        # Verify MCP name format
        assert MCPToolAdapter.is_mcp_tool(tool.mcp_name) is True
        server, name = MCPToolAdapter.parse_tool_name(tool.mcp_name)
        assert server == "my_server"
        assert name == "my_tool"


class TestAdapterIntegration:
    """Integration tests for adapter methods"""

    def test_full_workflow(self, sample_input_schema: Dict[str, Any]):
        """Test complete workflow: create -> convert -> parse"""
        # Create tool definition
        tool = MCPToolAdapter.create_tool_definition(
            server_name="test_server",
            tool_name="test_tool",
            description="Test tool description",
            input_schema=sample_input_schema,
            annotations={"key": "value"},
        )

        # Convert to Claude format
        claude_tool = MCPToolAdapter.to_claude_tool(tool)

        # Verify Claude format
        assert claude_tool["name"] == tool.mcp_name
        assert MCPToolAdapter.is_mcp_tool(claude_tool["name"]) is True

        # Parse MCP name
        server_name, original_name = MCPToolAdapter.parse_tool_name(claude_tool["name"])

        assert server_name == "test_server"
        assert original_name == "test_tool"

    def test_multiple_tools_consistency(self):
        """Test that multiple tools maintain consistency"""
        tools = []
        for i in range(5):
            tool = MCPToolAdapter.create_tool_definition(
                server_name=f"server_{i % 2}",
                tool_name=f"tool_{i}",
                description=f"Tool {i}",
                input_schema={"type": "object"},
            )
            tools.append(tool)

        # Verify all tools are unique
        mcp_names = [t.mcp_name for t in tools]
        assert len(set(mcp_names)) == len(mcp_names)

        # Verify all can be parsed
        for tool in tools:
            server, name = MCPToolAdapter.parse_tool_name(tool.mcp_name)
            assert server == tool.server_name
            assert name == tool.name


class TestEdgeCases:
    """Edge case tests"""

    def test_tool_name_with_numbers(self):
        """Test tool names containing numbers"""
        mcp_name = MCPToolAdapter.make_mcp_name("server1", "tool2_v3")
        assert mcp_name == "mcp__server1__tool2_v3"

        server, name = MCPToolAdapter.parse_tool_name(mcp_name)
        assert server == "server1"
        assert name == "tool2_v3"

    def test_tool_name_with_hyphens(self):
        """Test tool names containing hyphens"""
        mcp_name = MCPToolAdapter.make_mcp_name("my-server", "my-tool")
        assert mcp_name == "mcp__my-server__my-tool"

        server, name = MCPToolAdapter.parse_tool_name(mcp_name)
        assert server == "my-server"
        assert name == "my-tool"

    def test_tool_name_with_unicode(self):
        """Test tool names with unicode characters"""
        # Chinese characters in description
        tool = MCPToolAdapter.create_tool_definition(
            server_name="top",
            tool_name="query_station",
            description="查询场站信息",
            input_schema={"type": "object"},
        )

        assert tool.description == "查询场站信息"

        claude_tool = MCPToolAdapter.to_claude_tool(tool)
        assert claude_tool["description"] == "查询场站信息"

    def test_empty_input_schema(self):
        """Test with empty input schema"""
        tool = MCPToolAdapter.create_tool_definition(
            server_name="server",
            tool_name="tool",
            description="Test",
            input_schema={},
        )

        assert tool.input_schema == {}

        claude_tool = MCPToolAdapter.to_claude_tool(tool)
        assert claude_tool["input_schema"] == {}

    def test_very_long_tool_name(self):
        """Test with very long tool name"""
        long_name = "a" * 100
        mcp_name = MCPToolAdapter.make_mcp_name("server", long_name)

        assert long_name in mcp_name
        server, name = MCPToolAdapter.parse_tool_name(mcp_name)
        assert name == long_name
