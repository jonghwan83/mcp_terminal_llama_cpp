from app.policy import evaluate_tool_policy, get_tool_risk


def test_get_tool_risk_tiers() -> None:
    assert get_tool_risk("read_file") == "low"
    assert get_tool_risk("bash_exec") == "high"
    assert get_tool_risk("not_a_tool") == "unknown"


def test_policy_denies_unknown_tools() -> None:
    assert evaluate_tool_policy("not_a_tool", ask_permission=True) == "deny"
    assert evaluate_tool_policy("not_a_tool", ask_permission=False) == "deny"


def test_policy_requires_confirm_for_high_risk_when_prompting_enabled() -> None:
    assert evaluate_tool_policy("bash_exec", ask_permission=True) == "require_confirm"
    assert evaluate_tool_policy("write_file", ask_permission=True) == "require_confirm"


def test_policy_allows_low_risk_when_prompting_enabled() -> None:
    assert evaluate_tool_policy("read_file", ask_permission=True) == "allow"
    assert evaluate_tool_policy("search_code", ask_permission=True) == "allow"


def test_policy_allows_known_tools_when_prompting_disabled() -> None:
    assert evaluate_tool_policy("bash_exec", ask_permission=False) == "allow"
    assert evaluate_tool_policy("replace_in_file", ask_permission=False) == "allow"
