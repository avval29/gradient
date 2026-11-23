from gradient.rag import Retriever, chunk_text, load_corpus


def test_chunk_overlap_and_coverage():
    text = " ".join(f"w{i}" for i in range(200))
    chunks = chunk_text(text, size=50, overlap=10)
    assert len(chunks) > 1
    assert chunks[0].text.split()[-10:] == chunks[1].text.split()[:10]


def test_retrieves_relevant_chunk():
    r = Retriever(load_corpus("data/sample"))
    top, score = r.search("how do plants make food using sunlight", k=1)[0]
    assert "photosynthesis" in top.text.lower()
    assert score > 0.1
