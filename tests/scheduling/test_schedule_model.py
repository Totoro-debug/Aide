import json

import pytest

from aide.schedule.model import JobSchedule, ScheduleJob, ScheduleJobState

JOB_ID = "550e8400-e29b-41d4-a716-446655440000"


def test_schedule_job_round_trips_the_strict_persisted_shape() -> None:
    job = ScheduleJob(
        job_id=JOB_ID,
        message="Review the project.",
        title="Project review",
        schedule=JobSchedule(
            kind="cron",
            cron_expr="0 9 * * 1",
            timezone="Asia/Shanghai",
        ),
        created_at_ms=1_786_006_800_000,
        updated_at_ms=1_786_006_800_000,
    )

    persisted = job.to_dict()

    assert list(persisted) == [
        "job_id",
        "source",
        "message",
        "title",
        "schedule",
        "state",
        "created_at_ms",
        "updated_at_ms",
    ]
    assert persisted == {
        "job_id": JOB_ID,
        "source": "user",
        "message": "Review the project.",
        "title": "Project review",
        "schedule": {
            "kind": "cron",
            "at_time": None,
            "every_seconds": None,
            "cron_expr": "0 9 * * 1",
            "timezone": "Asia/Shanghai",
        },
        "state": {
            "last_finished_at_ms": None,
            "last_status": None,
            "last_error": None,
        },
        "created_at_ms": 1_786_006_800_000,
        "updated_at_ms": 1_786_006_800_000,
    }
    assert ScheduleJob.from_dict(json.loads(json.dumps(persisted))) == job


def test_schedule_job_derives_its_schedule_session_id() -> None:
    job = ScheduleJob(
        job_id=JOB_ID,
        message="Run this.",
        schedule=JobSchedule(kind="every", every_seconds=60),
        created_at_ms=1,
        updated_at_ms=1,
    )

    assert job.session_id == f"schedule_{JOB_ID}"


def test_schedule_job_accepts_the_reserved_dream_system_identity() -> None:
    job = ScheduleJob(
        job_id="dream",
        source="system",
        message="Internal Dream schedule.",
        schedule=JobSchedule(kind="cron", cron_expr="0 * * * *", timezone="UTC"),
        created_at_ms=1,
        updated_at_ms=1,
    )

    assert job.source == "system"
    assert job.job_id == "dream"
    assert job.title == "Dream"


def test_dream_title_is_fixed_even_when_its_message_is_unstable() -> None:
    job = _valid_job(
        job_id="dream",
        source="system",
        message="A changing internal message.",
    )

    assert job.title == "Dream"


def test_dream_rejects_a_noncanonical_explicit_title() -> None:
    with pytest.raises(ValueError, match="fixed"):
        _valid_job(job_id="dream", source="system", title="Derived from message")


def test_canonical_dream_schema_round_trips_the_fixed_title() -> None:
    job = _valid_job(job_id="dream", source="system", title="Dream")

    assert ScheduleJob.from_dict(job.to_dict()) == job


@pytest.mark.parametrize(
    ("job_id", "source"),
    [
        ("dream", "user"),
        (JOB_ID, "system"),
        ("unknown", "system"),
    ],
)
def test_schedule_job_rejects_invalid_system_identity(job_id: str, source: str) -> None:
    with pytest.raises(ValueError, match="job_id"):
        _valid_job(job_id=job_id, source=source)


@pytest.mark.parametrize(
    "schedule",
    [
        JobSchedule(kind="at", at_time="2026-08-07T12:00:00.000+08:00"),
        JobSchedule(kind="every", every_seconds=60),
        JobSchedule(kind="cron", cron_expr="0 9 * * 1", timezone="Asia/Shanghai"),
    ],
)
def test_schedule_values_use_explicit_nulls_for_unselected_fields(
    schedule: JobSchedule,
) -> None:
    assert set(schedule.to_dict()) == {
        "kind",
        "at_time",
        "every_seconds",
        "cron_expr",
        "timezone",
    }


def test_state_accepts_only_the_three_terminal_combinations() -> None:
    assert ScheduleJobState().to_dict() == {
        "last_finished_at_ms": None,
        "last_status": None,
        "last_error": None,
    }
    assert (
        ScheduleJobState(
            last_finished_at_ms=10,
            last_status="ok",
        ).to_dict()["last_error"]
        is None
    )
    assert (
        ScheduleJobState(
            last_finished_at_ms=10,
            last_status="error",
            last_error="The run failed.",
        ).to_dict()["last_status"]
        == "error"
    )


def _valid_job(**changes: object) -> ScheduleJob:
    values: dict[str, object] = {
        "job_id": JOB_ID,
        "message": "Run this.",
        "schedule": JobSchedule(kind="every", every_seconds=60),
        "created_at_ms": 10,
        "updated_at_ms": 10,
    }
    values.update(changes)
    return ScheduleJob(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("\n  \u300c  Project\t review  \u300d\nIgnore this line", "Project review"),
        ('"  Quoted title  "', "Quoted title"),
        ("'Single quoted title'", "Single quoted title"),
        ('"one-sided title', '"one-sided title'),
        ("a" * 59, "a" * 59),
        ("b" * 60, "b" * 60),
        ("c" * 61, "c" * 60),
        ("\U0001f600" * 61, "\U0001f600" * 60),
        ("e\u0301" * 31, "e\u0301" * 30),
    ],
)
def test_schedule_job_normalizes_explicit_title_and_truncates_by_code_point(
    title: str,
    expected: str,
) -> None:
    job = _valid_job(title=title)

    assert job.title == expected
    assert len(job.title) <= 60
    assert ScheduleJob.from_dict(job.to_dict()) == job


@pytest.mark.parametrize("title", ["", "\n\t", '""', "''", "\u300c\u300d"])
def test_schedule_job_rejects_explicit_title_empty_after_normalization(title: str) -> None:
    with pytest.raises(ValueError, match="title"):
        _valid_job(title=title)


def test_schedule_job_derives_title_from_the_first_nonempty_message_line() -> None:
    job = _valid_job(message="First\t line  \nSecond line")

    assert job.title == "First line"


@pytest.mark.parametrize(
    "value",
    [
        "550E8400-E29B-41D4-A716-446655440000",
        "550e8400e29b41d4a716446655440000",
        "550e8400-e29b-11d4-a716-446655440000",
    ],
)
def test_schedule_job_rejects_noncanonical_job_ids(value: str) -> None:
    with pytest.raises(ValueError, match="canonical UUID4"):
        _valid_job(job_id=value)


@pytest.mark.parametrize("message", ["", " Run this.", "Run this. ", "\t", "x" * 20_001])
def test_schedule_job_rejects_noncanonical_messages(message: str) -> None:
    with pytest.raises(ValueError, match="message"):
        _valid_job(message=message)


@pytest.mark.parametrize(
    "schedule",
    [
        JobSchedule(kind="at", at_time="2026-08-07T12:00:00.000+08:00"),
        JobSchedule(kind="every", every_seconds=1),
        JobSchedule(kind="cron", cron_expr="0 9 * * 1", timezone="Asia/Shanghai"),
    ],
)
def test_schedule_job_models_are_immutable(schedule: JobSchedule) -> None:
    job = _valid_job(schedule=schedule)

    with pytest.raises(AttributeError):
        job.message = "changed"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        schedule.kind = "at"  # type: ignore[misc]


@pytest.mark.parametrize(
    "schedule",
    [
        JobSchedule(kind="at", at_time="2026-08-07T12:00:00.000+08:00"),
        JobSchedule(kind="every", every_seconds=60),
        JobSchedule(kind="cron", cron_expr="0 9 * * 1", timezone="Asia/Shanghai"),
    ],
)
def test_schedule_job_round_trip_preserves_each_schedule_kind(schedule: JobSchedule) -> None:
    job = _valid_job(schedule=schedule)

    assert ScheduleJob.from_dict(job.to_dict()) == job


@pytest.mark.parametrize(
    "value",
    [
        "2026-08-07T12:00:00+08:00",
        "2026-08-07T12:00:00.00+08:00",
        "2026-08-07T12:00:00.000Z",
        "2026-08-07T12:00:00.000+0800",
        "2026-08-07T12:00:00.000-00:00",
        "2026-08-07 12:00:00.000+08:00",
    ],
)
def test_at_schedule_requires_canonical_offset_aware_milliseconds(value: str) -> None:
    with pytest.raises(ValueError, match="RFC 3339"):
        JobSchedule(kind="at", at_time=value)


@pytest.mark.parametrize(
    "every_seconds",
    [
        0,
        -1,
        True,
        1.5,
    ],
)
def test_every_schedule_requires_positive_integer(every_seconds: object) -> None:
    with pytest.raises(ValueError, match=r"positive|nonnegative"):
        JobSchedule(kind="every", every_seconds=every_seconds)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "values",
    [
        {"kind": "unknown"},
        {
            "kind": "at",
            "at_time": "2026-08-07T12:00:00.000+08:00",
            "every_seconds": 60,
        },
        {
            "kind": "at",
            "at_time": "2026-08-07T12:00:00.000+08:00",
            "timezone": "Asia/Shanghai",
        },
        {"kind": "every", "every_seconds": 60, "cron_expr": "0 9 * * 1"},
        {"kind": "cron", "cron_expr": "0 9 * * 1"},
        {
            "kind": "cron",
            "at_time": "2026-08-07T12:00:00.000+08:00",
            "cron_expr": "0 9 * * 1",
            "timezone": "Asia/Shanghai",
        },
    ],
)
def test_schedule_rejects_invalid_kind_field_combinations(values: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        JobSchedule(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("cron_expr", "timezone"),
    [
        ("0  9 * * 1", "Asia/Shanghai"),
        ("@daily", "Asia/Shanghai"),
        ("0 9 * * 1", "UTC+08:00"),
        ("0 9 * * 1", "Not/A_Timezone"),
    ],
)
def test_cron_schedule_requires_canonical_cron_and_valid_iana_timezone(
    cron_expr: str,
    timezone: str,
) -> None:
    with pytest.raises(ValueError):
        JobSchedule(kind="cron", cron_expr=cron_expr, timezone=timezone)


def test_cron_input_normalizes_whitespace_and_defaults_to_utc() -> None:
    assert JobSchedule.from_cron_input(" 0  9 * * 1 ") == JobSchedule.cron(
        "0 9 * * 1",
        "UTC",
    )


def test_cron_input_preserves_an_explicit_iana_timezone() -> None:
    assert JobSchedule.from_cron_input("0 9 * * *", "US/Eastern") == JobSchedule.cron(
        "0 9 * * *",
        "US/Eastern",
    )


def test_at_input_normalizes_timezone_aware_iso_values() -> None:
    assert JobSchedule.from_at_input("2020-01-02T03:04:05.123456Z") == JobSchedule.at(
        "2020-01-02T03:04:05.123+00:00"
    )


@pytest.mark.parametrize("value", ["2026-08-07T13:00:00", 1])
def test_at_input_rejects_naive_or_non_string_values(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        JobSchedule.from_at_input(value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cron_expr", "0  9 * * 1"),
        ("timezone", None),
    ],
)
def test_schedule_job_from_dict_does_not_normalize_cron_input(
    field: str,
    value: object,
) -> None:
    document = _valid_job(schedule=JobSchedule.cron("0 9 * * 1", "Asia/Shanghai")).to_dict()
    schedule = document["schedule"]
    assert isinstance(schedule, dict)
    schedule[field] = value

    with pytest.raises(ValueError):
        ScheduleJob.from_dict(document)


@pytest.mark.parametrize(
    "state",
    [
        {"last_finished_at_ms": 1, "last_status": None},
        {"last_finished_at_ms": True, "last_status": "ok"},
        {"last_finished_at_ms": None, "last_status": "ok"},
        {"last_finished_at_ms": 1, "last_status": "unknown"},
        {"last_finished_at_ms": 1, "last_status": "ok", "last_error": "failed"},
        {"last_finished_at_ms": 1, "last_status": "error"},
        {"last_finished_at_ms": 1, "last_status": "error", "last_error": ""},
        {"last_finished_at_ms": 1, "last_status": "error", "last_error": 1},
    ],
)
def test_state_rejects_invalid_combinations(state: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ScheduleJobState(**state)  # type: ignore[arg-type]


def test_schedule_job_rejects_updated_timestamp_before_creation() -> None:
    with pytest.raises(ValueError, match="updated_at_ms"):
        _valid_job(created_at_ms=10, updated_at_ms=9)


def test_schedule_job_rejects_changed_timestamp_without_a_terminal_state() -> None:
    with pytest.raises(ValueError, match="never-run"):
        _valid_job(created_at_ms=10, updated_at_ms=11)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"source": "external"}, "source"),
        ({"created_at_ms": True}, "created_at_ms"),
        ({"created_at_ms": -1}, "created_at_ms"),
        ({"updated_at_ms": True}, "updated_at_ms"),
    ],
)
def test_schedule_job_rejects_invalid_source_and_timestamps(
    changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _valid_job(**changes)


def test_schedule_job_from_dict_rejects_unknown_or_missing_fields() -> None:
    document = _valid_job().to_dict()

    with pytest.raises(ValueError, match="persisted schema"):
        ScheduleJob.from_dict({**document, "extra": True})
    with pytest.raises(ValueError, match="persisted schema"):
        ScheduleJob.from_dict({key: value for key, value in document.items() if key != "state"})


@pytest.mark.parametrize("title", [None, "  Canonical ", "x" * 61])
def test_schedule_job_from_dict_rejects_noncanonical_new_schema_title(title: object) -> None:
    document = _valid_job(title="Canonical").to_dict()
    document["title"] = title

    with pytest.raises(ValueError, match="title"):
        ScheduleJob.from_dict(document)


@pytest.mark.parametrize(
    ("container", "field", "remove"),
    [
        ("schedule", "extra", False),
        ("schedule", "timezone", True),
        ("state", "extra", False),
        ("state", "last_error", True),
    ],
)
def test_schedule_job_from_dict_rejects_nonexact_nested_fields(
    container: str,
    field: str,
    remove: bool,
) -> None:
    document = _valid_job().to_dict()
    nested = document[container]
    assert isinstance(nested, dict)
    if remove:
        nested.pop(field)
    else:
        nested[field] = None

    with pytest.raises(ValueError, match="fields"):
        ScheduleJob.from_dict(document)
