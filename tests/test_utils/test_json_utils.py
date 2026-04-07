import pytest
from src.customer_helpdesk.utils.json_utils import safe_json_parse, safe_json_dumps


class TestSafeJsonParse:
    def test_valid_json_string(self):
        result = safe_json_parse('{"key": "value"}')
        assert result == {"key": "value"}

    def test_invalid_json_returns_fallback(self):
        result = safe_json_parse("not valid json")
        assert result == {}

    def test_empty_string_returns_fallback(self):
        result = safe_json_parse("")
        assert result == {}

    def test_dict_input_returned_as_is(self):
        input_dict = {"already": "dict"}
        result = safe_json_parse(input_dict)
        assert result is input_dict

    def test_none_input_returns_fallback(self):
        result = safe_json_parse(None)
        assert result == {}

    def test_custom_fallback_on_invalid(self):
        result = safe_json_parse("invalid", fallback={"custom": "fallback"})
        assert result == {"custom": "fallback"}

    def test_none_as_custom_fallback(self):
        result = safe_json_parse("invalid", fallback=None)
        assert result is None


class TestSafeJsonDumps:
    def test_serializable_object(self):
        result = safe_json_dumps({"key": "value"})
        assert result == '{"key": "value"}'

    def test_non_serializable_object_uses_default_str(self):
        class NotSerializable:
            pass

        result = safe_json_dumps(NotSerializable())
        assert "NotSerializable object at" in result

    def test_custom_fallback_not_needed_with_default_str(self):
        class NotSerializable:
            pass

        result = safe_json_dumps(NotSerializable(), fallback='{"error": true}')
        assert "NotSerializable object at" in result

    def test_list_with_non_serializable(self):
        result = safe_json_dumps([1, 2, {"key": "value"}])
        assert result == '[1, 2, {"key": "value"}]'

    def test_primitive_types(self):
        assert safe_json_dumps("string") == '"string"'
        assert safe_json_dumps(123) == "123"
        assert safe_json_dumps(45.67) == "45.67"
        assert safe_json_dumps(True) == "true"
        assert safe_json_dumps(None) == "null"
