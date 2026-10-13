import json
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from organize import Config
from organize.filters.classify import Classify, contains, cosine
from organize.resource import Resource


@pytest.fixture
def classifier():
    return Classify(
        categories=[
            {
                "id": "creator",
                "folder": "运营/创作者",
                "strong": ["创作者运营", "达人孵化"],
                "any": ["音乐人", "激励"],
                "exclude": ["员工招聘"],
            },
            {"id": "rag", "folder": "技术/RAG", "strong": ["RAG", "向量数据库"]},
        ]
    )


def decide(classifier, relative):
    return classifier.classify(
        classifier.context(Path("/incoming") / relative, Path("/incoming"))
    )


def test_generic_title_inherits_parent(classifier):
    result = decide(classifier, "创作者运营/课程一/01.mp4")
    assert result["category_id"] == "creator"
    assert result["context"]["title"] == ""
    assert result["relative_parent"] == "创作者运营/课程一"


def test_scan_root_and_depth_are_respected(classifier):
    context = classifier.context(Path("/RAG/incoming/01.mp4"), Path("/RAG/incoming"))
    assert context["parents"] == []
    assert classifier.classify(context)["status"] == "review"
    classifier.include_root = True
    assert decide(classifier, "01.mp4")["status"] == "review"
    classifier.depth = 1
    assert decide(classifier, "RAG/杂项/01.mp4")["status"] == "review"
    with pytest.raises(ValueError):
        classifier.context(Path("/outside/file.mp4"), Path("/incoming"))


def test_root_can_be_explicitly_included(classifier):
    classifier.include_root = True
    context = classifier.context(Path("/RAG/01.mp4"), Path("/RAG"))
    assert classifier.classify(context)["category_id"] == "rag"


def test_cross_source_combination(classifier):
    assert decide(classifier, "音乐人/第三课_激励.mp4")["category_id"] == "creator"


def test_clear_title_directory_conflict_is_review(classifier):
    result = decide(classifier, "创作者运营/向量数据库.mp4")
    assert result["status"] == "review"
    assert result["reason"] == "title_directory_conflict"


def test_exclude_is_hard_constraint(classifier):
    classifier = Classify(
        categories=[c.model_dump() for c in classifier.categories],
        model_api={"base_url": "https://example.invalid/v1", "llm_model": "test"},
    )
    with patch.object(
        Classify,
        "_post",
        return_value={
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "category_id": "creator",
                                "needs_review": False,
                                "evidence": ["创作者运营"],
                            }
                        )
                    }
                }
            ]
        },
    ):
        result = decide(classifier, "员工招聘/创作者运营.mp4")
    assert result["status"] == "review"
    assert all(x["id"] != "creator" for x in result["candidates"])


def test_alias_case_and_ascii_boundary(classifier):
    classifier.aliases = {"知识库搭建": "RAG"}
    assert decide(classifier, "知识库搭建.mp4")["category_id"] == "rag"
    assert decide(classifier, "ＲＡＧ_1080P.mp4")["category_id"] == "rag"
    assert not contains("dragon tutorial", "rag")


@pytest.mark.parametrize(
    "folder", ["../escape", "/absolute", "C:/escape", "a/../../b", "a\\..\\b", ".", ""]
)
def test_invalid_destination_rejected(folder):
    with pytest.raises(ValidationError):
        Classify(categories=[{"id": "a", "folder": folder}])


def test_duplicate_ids_and_empty_keywords_rejected():
    with pytest.raises(ValidationError):
        Classify(categories=[{"id": "a", "folder": "a"}, {"id": "a", "folder": "b"}])
    with pytest.raises(ValidationError):
        Classify(categories=[{"id": "a", "folder": "a", "strong": [" "]}])


def test_semantic_acceptance_and_model_cache(classifier):
    classifier = Classify(
        categories=[c.model_dump() for c in classifier.categories],
        model_api={"base_url": "https://example.invalid/v1", "embedding_model": "test"},
    )

    def embedding(route, body):
        return {
            "data": [
                {
                    "embedding": [1, 0]
                    if "creator" in body["input"] or "持续投稿" in body["input"]
                    else [0, 1]
                }
            ]
        }

    with patch.object(Classify, "_post", side_effect=embedding) as post:
        result = decide(classifier, "如何持续投稿.mp4")
        assert result["category_id"] == "creator"
        assert result["stage"] == "semantic"
        decide(classifier, "如何持续投稿.mp4")
        assert post.call_count == 3


def test_semantic_tie_and_conflict_do_not_auto_accept(classifier):
    classifier = Classify(
        categories=[c.model_dump() for c in classifier.categories],
        model_api={"base_url": "https://example.invalid/v1", "embedding_model": "test"},
    )
    with patch.object(
        Classify, "_post", return_value={"data": [{"embedding": [1, 0]}]}
    ):
        assert decide(classifier, "知识讲解.mp4")["status"] == "review"
        assert decide(classifier, "创作者运营/向量数据库.mp4")["status"] == "review"


@pytest.mark.parametrize(
    "decision",
    [
        {"category_id": "unknown", "needs_review": False, "evidence": ["教程"]},
        {"category_id": "creator", "needs_review": False, "evidence": ["不存在的词"]},
        {"category_id": "creator", "needs_review": True, "evidence": ["教程"]},
        {"category_id": "creator", "needs_review": False, "evidence": []},
        [],
    ],
)
def test_invalid_llm_decisions_are_review(classifier, decision):
    classifier = Classify(
        categories=[c.model_dump() for c in classifier.categories],
        model_api={"base_url": "https://example.invalid/v1", "llm_model": "test"},
    )
    with patch.object(
        Classify,
        "_post",
        return_value={"choices": [{"message": {"content": json.dumps(decision)}}]},
    ):
        assert decide(classifier, "教程.mp4")["status"] == "review"


def test_llm_conflict_keeps_both_candidates(classifier):
    classifier = Classify(
        categories=[c.model_dump() for c in classifier.categories],
        model_api={
            "base_url": "https://example.invalid/v1",
            "llm_model": "test",
            "top_k": 1,
        },
    )
    with patch.object(
        Classify,
        "_post",
        return_value={
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "category_id": "rag",
                                "needs_review": False,
                                "evidence": ["向量数据库"],
                            }
                        )
                    }
                }
            ]
        },
    ) as post:
        result = decide(classifier, "创作者运营/向量数据库.mp4")
        assert result["category_id"] == "rag"
        assert result["stage"] == "llm"
        prompt = json.loads(post.call_args.args[1]["messages"][1]["content"])
        assert len(prompt["categories"]) == 2


def test_model_outage_and_short_circuit(classifier):
    classifier = Classify(
        categories=[c.model_dump() for c in classifier.categories],
        model_api={"base_url": "https://example.invalid/v1", "llm_model": "test"},
    )
    with patch.object(Classify, "_post", side_effect=TimeoutError):
        assert (
            decide(classifier, "课程讲解.mp4")["reason"]
            == "model_unavailable_or_invalid"
        )
    with patch.object(Classify, "_post") as post:
        assert decide(classifier, "RAG.mp4")["category_id"] == "rag"
        assert decide(classifier, "01.mp4")["status"] == "review"
        post.assert_not_called()


def test_pipeline_review_and_metadata(classifier, testoutput):
    resource = Resource(path=Path("/incoming/01.mp4"), basedir=Path("/incoming"))
    assert not classifier.pipeline(resource, testoutput)
    assert json.loads(resource.vars["classify"]["json"])["source"] == "/incoming/01.mp4"
    classifier.match = "review"
    assert classifier.pipeline(resource, testoutput)


@pytest.mark.parametrize(
    "a,b", [([], []), ([0, 0], [1, 1]), ([1], [1, 2]), ([float("nan")], [1])]
)
def test_invalid_embeddings(a, b):
    with pytest.raises(ValueError):
        cosine(a, b)


def test_simulation_and_actual_move_preserve_course_and_conflicts(tmp_path):
    incoming = tmp_path / "incoming"
    course = incoming / "RAG课程"
    course.mkdir(parents=True)
    source = course / "01.mp4"
    source.write_bytes(b"video")
    other = incoming / "02.mp4"
    other.write_bytes(b"unknown")
    library = tmp_path / "library"
    config = Config.from_string(f'''
    rules:
      - locations: {incoming.as_posix()}
        subfolders: true
        filters:
          - extension: mp4
          - classify:
              categories:
                - id: rag
                  folder: 技术/RAG
                  strong: [RAG课程]
        actions:
          - move:
              dest: "{library.as_posix()}/{{classify.folder}}/{{classify.relative_parent}}/"
              on_conflict: skip
    ''')
    config.execute(simulate=True)
    assert source.exists() and not library.exists()
    dest = library / "技术/RAG/RAG课程/01.mp4"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"existing")
    config.execute(simulate=False)
    assert source.exists() and dest.read_bytes() == b"existing"
    dest.unlink()
    config.execute(simulate=False)
    assert not source.exists() and dest.read_bytes() == b"video"
    assert other.exists()


def test_remote_auth_url_and_timeout(classifier, monkeypatch):
    from unittest.mock import MagicMock

    classifier = Classify(
        categories=[c.model_dump() for c in classifier.categories],
        model_api={
            "base_url": "https://example.invalid/v1/",
            "llm_model": "test",
            "timeout": 12,
        },
    )
    monkeypatch.setenv("NAS_MODEL_API_KEY", "test-only-not-a-real-key")
    response = MagicMock()
    response.__enter__.return_value.read.return_value = b'{"choices": []}'
    opener = MagicMock()
    opener.open.return_value = response
    with patch("organize.filters.classify.build_opener", return_value=opener):
        assert classifier._post("/chat/completions", {"model": "test"}) == {
            "choices": []
        }
    request = opener.open.call_args.args[0]
    assert request.full_url == "https://example.invalid/v1/chat/completions"
    assert request.get_header("Authorization") == "Bearer test-only-not-a-real-key"
    assert opener.open.call_args.kwargs["timeout"] == 12


def test_missing_key_fails_closed_without_network(classifier, monkeypatch):
    monkeypatch.delenv("NAS_MODEL_API_KEY", raising=False)
    classifier = Classify(
        categories=[c.model_dump() for c in classifier.categories],
        model_api={"base_url": "https://example.invalid/v1", "llm_model": "test"},
    )
    with patch("organize.filters.classify.build_opener") as opener:
        assert decide(classifier, "教程.mp4")["status"] == "review"
        opener.assert_not_called()


def test_embeddings_outage_can_fall_back_to_chat(classifier):
    classifier = Classify(
        categories=[c.model_dump() for c in classifier.categories],
        model_api={
            "base_url": "https://example.invalid/v1",
            "llm_model": "test",
            "embedding_model": "test",
            "json_mode": False,
        },
    )

    def response(route, body):
        if route == "/embeddings":
            raise TimeoutError()
        assert route == "/chat/completions"
        assert "response_format" not in body
        assert body["max_tokens"] == 1024
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "category_id": "creator",
                                "needs_review": False,
                                "evidence": ["教程"],
                            }
                        )
                    }
                }
            ]
        }

    with patch.object(Classify, "_post", side_effect=response):
        assert decide(classifier, "教程.mp4")["stage"] == "llm"


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.com",
        "https://user:password@example.com",
        "https://example.com?key=value",
        "https://example.com#fragment",
    ],
)
def test_invalid_api_urls(url):
    from organize.filters.classify import ModelAPI

    with pytest.raises(ValidationError):
        ModelAPI(base_url=url)


def test_redirects_do_not_forward_credentials():
    from organize.filters.classify import NoRedirect

    with pytest.raises(ValueError):
        NoRedirect().redirect_request(None, None, 302, "", {}, "https://other.invalid")


def test_path_traversal_and_symlink_outside_root(classifier, tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(ValueError):
        classifier.context(root / ".." / "other.mp4", root)
    outside = tmp_path / "outside.mp4"
    outside.touch()
    link = root / "link.mp4"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("Symlinks unavailable on this platform")
    with pytest.raises(ValueError):
        classifier.context(link, root)
