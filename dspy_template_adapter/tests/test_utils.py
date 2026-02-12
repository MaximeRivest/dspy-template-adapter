"""Test parsing utils."""
from dspy_template_adapter.template_adapter import _parse_func_kwargs

def test_parse_arguments():
    # Simple
    assert _parse_func_kwargs("a='b', c='d'") == {"a": "b", "c": "d"}
    # Quoted comma
    assert _parse_func_kwargs("styles='json,yaml'") == {"styles": "json,yaml"}
    # Double quotes
    assert _parse_func_kwargs('a="b, c"') == {"a": "b, c"}
    # Mixed and spaces
    assert _parse_func_kwargs(" format = 'json' ,  other= \"val\" ") == {"format": "json", "other": "val"}
    # Empty
    assert _parse_func_kwargs("") == {}
