from pathlib import Path

import pytest

from investigation_engine.domain import Document
from investigation_engine.ingestion.parsers import (
    SUPPORTED_EXTENSIONS,
    ParseError,
    UnsupportedFormatError,
    parse_bytes,
    parse_path,
)


def parse(filename: str, content: str) -> list[Document]:
    return parse_bytes(filename, content.encode("utf-8"))


def test_plain_text_becomes_one_document_named_after_the_file() -> None:
    documents = parse("notes.txt", "Hello world\n")

    assert documents == [
        Document(
            id="notes.txt",
            text="Hello world",
            title="notes",
            source="notes.txt",
            metadata={"format": "txt"},
        )
    ]


def test_byte_order_mark_and_windows_line_endings_are_normalised() -> None:
    documents = parse_bytes("a.txt", b"\xef\xbb\xbfHello\r\nworld")

    assert documents[0].text == "Hello\nworld"


def test_undecodable_bytes_become_replacement_characters() -> None:
    documents = parse_bytes("a.txt", b"caf\xe9")

    assert documents[0].text == "caf�"


def test_blank_file_yields_no_documents() -> None:
    assert parse("empty.txt", "  \n\n ") == []


def test_extension_is_case_insensitive_and_directories_are_dropped() -> None:
    documents = parse("reports/NOTES.TXT", "x")

    assert (documents[0].id, documents[0].source) == ("NOTES.TXT", "NOTES.TXT")


def test_unsupported_extension_is_rejected_with_the_extension_named() -> None:
    with pytest.raises(UnsupportedFormatError, match=r"\.png"):
        parse_bytes("image.png", b"")


def test_supported_extensions_list_every_registered_format() -> None:
    assert set(SUPPORTED_EXTENSIONS) == {".txt", ".md", ".html", ".htm", ".json", ".jsonl", ".csv"}


def test_markdown_title_comes_from_the_first_top_level_heading() -> None:
    documents = parse("outage.md", "# Payment outage\n\nRoot cause: bad deploy.\n")

    assert documents[0].title == "Payment outage"
    assert documents[0].text == "# Payment outage\n\nRoot cause: bad deploy."
    assert documents[0].metadata == {"format": "md"}


def test_markdown_without_a_top_level_heading_falls_back_to_the_file_name() -> None:
    documents = parse("runbook.md", "## Steps\n\nRestart the service.\n")

    assert documents[0].title == "runbook"


def test_html_text_excludes_tags_scripts_and_styles() -> None:
    page = (
        "<html><head><title>Outage report</title><style>p{color:red}</style></head>"
        "<body><h1>Outage</h1><p>Payments &amp; refunds failed.</p>"
        "<script>alert(1)</script></body></html>"
    )

    documents = parse("report.html", page)

    assert documents[0].title == "Outage report"
    assert documents[0].text == "Outage\nPayments & refunds failed."
    assert documents[0].metadata == {"format": "html"}


def test_html_title_falls_back_to_the_first_heading() -> None:
    documents = parse("page.htm", "<body><h1>First</h1><h1>Second</h1><p>Body</p></body>")

    assert documents[0].title == "First"
    assert documents[0].metadata == {"format": "html"}


def test_html_meta_tags_become_metadata() -> None:
    page = (
        '<head><meta charset="utf-8"><meta name="Date" content="2024-03-12">'
        '<meta name="author" content="SRE team"></head><body><p>Body</p></body>'
    )

    documents = parse("page.html", page)

    assert documents[0].metadata == {"format": "html", "date": "2024-03-12", "author": "SRE team"}


def test_json_object_becomes_one_document_with_scalar_metadata() -> None:
    content = '{"id": "inc-1", "title": "Incident 1", "text": "Body", "service": "payments"}'

    documents = parse("data.json", content)

    assert documents == [
        Document(
            id="data.json#inc-1",
            text="Body\nid: inc-1; service: payments",
            title="Incident 1",
            source="data.json",
            metadata={"format": "json", "service": "payments"},
        )
    ]


def test_json_array_numbers_records_that_have_no_id() -> None:
    documents = parse("data.json", '[{"text": "first"}, {"text": "second"}]')

    assert [document.id for document in documents] == ["data.json#0", "data.json#1"]
    assert [document.text for document in documents] == ["first", "second"]


def test_a_record_with_an_id_but_no_other_fields_gets_a_line_with_just_the_id() -> None:
    documents = parse("data.json", '[{"id": "x-1", "text": "Body"}]')

    assert documents[0].text == "Body\nid: x-1"


def test_record_fields_are_added_to_the_text_so_they_can_be_searched_and_read() -> None:
    documents = parse(
        "data.jsonl", '{"id": "r1", "text": "Body", "risk": "low", "date": "2025-03-04"}'
    )

    assert documents[0].text == "Body\nid: r1; risk: low; date: 2025-03-04"


def test_a_record_without_text_stays_skipped_even_when_it_has_fields() -> None:
    assert parse("data.json", '[{"id": "x", "text": "", "risk": "low"}]') == []


def test_json_metadata_keeps_only_non_empty_scalars() -> None:
    content = (
        '[{"text": "t", "count": 3, "ratio": 0.5, "flag": true, '
        '"tags": ["a"], "owner": null, "note": ""}]'
    )

    documents = parse("data.json", content)

    assert documents[0].metadata == {
        "format": "json",
        "count": "3",
        "ratio": "0.5",
        "flag": "true",
    }
    assert documents[0].text == "t\ncount: 3; ratio: 0.5; flag: true"


def test_json_records_with_empty_text_are_skipped() -> None:
    documents = parse("data.json", '[{"text": ""}, {"text": "kept"}]')

    assert [document.id for document in documents] == ["data.json#1"]


def test_json_syntax_error_is_reported_with_the_file_name() -> None:
    with pytest.raises(ParseError, match=r"data\.json.*invalid JSON"):
        parse("data.json", '{"text": ')


def test_json_record_without_text_is_reported_with_its_position() -> None:
    with pytest.raises(ParseError, match="record 0"):
        parse("data.json", '[{"title": "no text"}]')


def test_json_top_level_scalar_is_rejected() -> None:
    with pytest.raises(ParseError, match="not an object"):
        parse("data.json", "5")


def test_jsonl_reads_one_document_per_line_and_ignores_blank_lines() -> None:
    content = '{"id": "a", "text": "first"}\n\n{"id": "b", "text": "second"}\n'

    documents = parse("lines.jsonl", content)

    assert [document.id for document in documents] == ["lines.jsonl#a", "lines.jsonl#b"]
    assert documents[0].metadata == {"format": "jsonl"}


def test_jsonl_syntax_error_names_the_line() -> None:
    with pytest.raises(ParseError, match="line 2"):
        parse("lines.jsonl", '{"text": "ok"}\nnot json\n')


def test_csv_rows_become_documents_and_empty_cells_are_left_out_of_metadata() -> None:
    content = "id,title,text,date\n1,Outage,Payments failed,2024-03-12\n2,Refunds,Refunds late,\n"

    documents = parse("incidents.csv", content)

    assert [document.id for document in documents] == ["incidents.csv#1", "incidents.csv#2"]
    assert documents[0].title == "Outage"
    assert documents[0].metadata == {"format": "csv", "date": "2024-03-12"}
    assert documents[1].metadata == {"format": "csv"}
    assert documents[0].text == "Payments failed\nid: 1; date: 2024-03-12"
    assert documents[1].text == "Refunds late\nid: 2"


def test_csv_keeps_newlines_inside_quoted_cells() -> None:
    documents = parse("notes.csv", 'text\n"line one\nline two"\n')

    assert documents[0].text == "line one\nline two"
    assert documents[0].id == "notes.csv#0"


def test_csv_without_a_text_column_is_rejected() -> None:
    with pytest.raises(ParseError, match="text"):
        parse("notes.csv", "title,body\na,b\n")


def test_parse_path_reads_the_file_and_uses_its_name_as_source(tmp_path: Path) -> None:
    path = tmp_path / "incident.txt"
    path.write_text("Disk full on db-2", encoding="utf-8")

    documents = parse_path(path)

    assert [(document.id, document.text) for document in documents] == [
        ("incident.txt", "Disk full on db-2")
    ]
