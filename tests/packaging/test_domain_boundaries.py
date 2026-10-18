"""Concrete tool adapters and application ports satisfy the domain ToolSpec.

Import direction is checked separately by lint-imports.
"""

def test_tool_base_satisfies_tool_spec():
    from opencollab.adapters.tools.base import Tool
    from opencollab.domain.tools import ToolSpec

    instance: ToolSpec = Tool()
    assert isinstance(instance.name, str)
    assert isinstance(instance.description, str)
    assert isinstance(instance.parameters, dict)
    assert callable(instance.to_openai_schema)

def test_tool_port_carries_tool_spec_schema_surface():
    from opencollab.application.ports import ToolPort
    from opencollab.domain.tools import ToolSpec

    required_annotations = {"name", "description", "parameters"}
    assert required_annotations.issubset(ToolSpec.__annotations__)
    assert required_annotations.issubset(ToolPort.__annotations__)
    assert callable(ToolSpec.__dict__["to_openai_schema"])
    assert callable(ToolPort.__dict__["to_openai_schema"])
