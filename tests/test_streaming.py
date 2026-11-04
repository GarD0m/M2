from llm5.data import documents
from llm5.tokenizer import sample_texts


def test_documents_does_not_shuffle(monkeypatch):
    import datasets
    calls = []
    class Source:
        def shuffle(self, **kwargs):
            raise AssertionError("shuffle would enable unwanted shard prefetch")
        def __iter__(self):
            yield {"content": "a training document"}
    def load(*args, **kwargs):
        calls.append(kwargs)
        return Source()
    monkeypatch.setattr(datasets, "load_dataset", load)
    assert list(documents("general", "train", subset_index=0)) == ["a training document"]
    assert len(calls) == 1
    assert calls[0]["streaming"] is True
    assert calls[0]["columns"] == ["content"]
    assert calls[0]["batch_size"] == 16


def test_tokenizer_has_only_one_active_reader(monkeypatch):
    active = peak = 0
    def source(*args, **kwargs):
        nonlocal active, peak
        active += 1
        peak = max(active, peak)
        try:
            while True:
                yield "x" * 100
        finally:
            active -= 1
    monkeypatch.setattr("llm5.tokenizer.documents", source)
    output = list(sample_texts(1000, 10000))
    assert peak == 1 and active == 0
    assert sum(map(len, output)) == 10000
