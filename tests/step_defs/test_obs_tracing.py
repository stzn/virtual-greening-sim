"""LangSmith によるトレース。外部へは何も送らない（設定の解決と、各ターンの実行情報だけを検証する）。"""

from typing import Any

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from tests.step_defs.conftest import SimCtx
from vmgs.config import Settings
from vmgs.obs.tracing import TracingConfig, apply_tracing, resolve_tracing

scenarios("obs/tracing.feature")

SECRET = "ls-secret-abc123"


class TraceCtx:
    def __init__(self) -> None:
        self.settings_kwargs: dict[str, Any] = {}
        self.enable = False
        self.project: str | None = None
        self.cfg: TracingConfig | None = None

    def result(self) -> TracingConfig:
        assert self.cfg is not None
        return self.cfg


@pytest.fixture
def tr(monkeypatch: pytest.MonkeyPatch) -> TraceCtx:
    for name in ("LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT"):
        monkeypatch.delenv(name, raising=False)  # 実際のシェル環境を読まない
    return TraceCtx()


# ---------------- 設定の解決 ----------------
@given("トレース関連の設定が何も無い")
def _g_nothing(tr: TraceCtx) -> None:
    tr.settings_kwargs = {}


@given("LANGSMITH_TRACING が true でAPIキーが無い")
def _g_env_on_no_key(tr: TraceCtx) -> None:
    tr.settings_kwargs["langsmith_tracing"] = True


@given("APIキーが設定されている")
def _g_key(tr: TraceCtx) -> None:
    tr.settings_kwargs["langsmith_api_key"] = SECRET


@given("トレースの有効化が明示的に指定されている")
def _g_enable(tr: TraceCtx) -> None:
    tr.enable = True


@given(parsers.parse('プロジェクト名として "{p}" が指定されている'))
def _g_project(tr: TraceCtx, p: str) -> None:
    tr.project = p


@when("トレースの設定を解決する")
def _w_resolve(tr: TraceCtx) -> None:
    settings = Settings(_env_file=None, **tr.settings_kwargs)  # type: ignore[call-arg]
    tr.cfg = resolve_tracing(settings, enable=tr.enable, project=tr.project)


@then("トレースは無効である")
def _t_disabled(tr: TraceCtx) -> None:
    assert tr.result().enabled is False


@then("トレースは有効である")
def _t_enabled(tr: TraceCtx) -> None:
    assert tr.result().enabled is True


@then("トレース用の環境変数は何も設定されない")
def _t_no_env(tr: TraceCtx) -> None:
    cfg = tr.result()
    env: dict[str, str] = {}
    apply_tracing(cfg, env)  # 無効なら何も設定されない
    assert cfg.env == {}
    assert env == {}


@then(parsers.parse('無効の理由は "{text}" を含む'))
def _t_reason(tr: TraceCtx, text: str) -> None:
    reason = tr.result().reason
    assert reason is not None
    assert text in reason


@then(parsers.parse("トレース用の環境変数に {key}={value} が含まれる"))
def _t_env(tr: TraceCtx, key: str, value: str) -> None:
    cfg = tr.result()
    env: dict[str, str] = {}
    apply_tracing(cfg, env)
    assert env[key] == value


@then(parsers.parse('警告に "{a}" と "{b}" が含まれる'))
def _t_warning(tr: TraceCtx, a: str, b: str) -> None:
    text = "\n".join(tr.result().warnings)
    assert a in text
    assert b in text


@then("警告・理由・説明の文字列にAPIキーの値が含まれない")
def _t_no_secret(tr: TraceCtx) -> None:
    cfg = tr.result()
    shown = "\n".join([*cfg.warnings, cfg.reason or "", cfg.describe()])
    assert SECRET not in shown


# ---------------- 各ターンの実行情報 ----------------
@then(parsers.parse('実行情報の run_name は 1時間目から順に "{a}" "{b}" "{c}" である'))
def _t_run_names(sc: SimCtx, a: str, b: str, c: str) -> None:
    assert [rc["run_name"] for rc in sc.run_configs] == [a, b, c]


@then(parsers.re(r"^実行情報のメタデータに (?P<key>\w+)=(?P<value>\S+) が含まれる$"))
def _t_meta_kv(sc: SimCtx, key: str, value: str) -> None:
    assert all(str(rc["metadata"][key]) == value for rc in sc.run_configs)


@then(parsers.re(r"^実行情報のメタデータに (?P<key>\w+) が含まれる$"))
def _t_meta_key(sc: SimCtx, key: str) -> None:
    assert all(key in rc["metadata"] for rc in sc.run_configs)


@then("実行情報のメタデータに step と sim_time が含まれる")
def _t_meta_step(sc: SimCtx) -> None:
    for i, rc in enumerate(sc.run_configs, start=1):
        assert rc["metadata"]["step"] == i
        assert rc["metadata"]["sim_time"].startswith("2025-08-30T")


@then(parsers.parse('実行情報のタグに "{tag}" が含まれる'))
def _t_tag(sc: SimCtx, tag: str) -> None:
    assert all(tag in rc["tags"] for rc in sc.run_configs)
