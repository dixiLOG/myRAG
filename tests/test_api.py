from fastapi.testclient import TestClient

from myrag.api.app import create_app


def test_api_query_and_benchmark_endpoints() -> None:
    client = TestClient(create_app())
    health = client.get('/health')
    assert health.status_code == 200
    query = client.post('/query', json={'question': 'Linux shell 学习里提到的九大分类包括哪些方向？'})
    assert query.status_code == 200
    payload = query.json()
    assert payload['citations']
    benchmark = client.post('/benchmark/run')
    assert benchmark.status_code == 200
    assert benchmark.json()['results']
