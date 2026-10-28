from investigation_engine.llm.types import Usage


def test_adding_usages_sums_each_kind_of_token() -> None:
    assert Usage(5, 2) + Usage(3, 1) == Usage(8, 3)


def test_an_empty_usage_is_the_neutral_element() -> None:
    assert Usage(5, 2) + Usage() == Usage(5, 2)
