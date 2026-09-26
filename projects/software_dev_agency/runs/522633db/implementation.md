```python
def build_solution(user_input: str) -> str:
    if not user_input.strip():
        raise ValueError("Input is required")
    return f"Processed: {user_input.strip()}"
```

```python
def test_build_solution():
    assert build_solution("demo") == "Processed: demo"
```
