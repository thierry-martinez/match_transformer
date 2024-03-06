# match_transformer

Rewrite `match` into `if` to make code compatible with Python <3.10.

This small refactoring tool uses
[LibCST](https://github.com/Instagram/LibCST) to rewrite Python
modules while preserving the code layout and comments as much as
possible in order to replace `match` blocks into `if-elif-else`
blocks, so as to make the code compatible with Python <3.10 (`match`
was introduced by [PEP 634](https://peps.python.org/pep-0634/) with
Python 3.10).

## Usage

```
$ pip install -r requirements.txt
$ python match_transformer.py <original_module.py >rewritten_module.py
```

## Examples

The following code

```python
match e:
    case "A":
        print("Hello!")
    case "B":
        print("Hi!")
```

is rewritten as

```python
# match e:
#     case "A":
#         print("Hello!")
#     case "B":
#         print("Hi!")
if e == "A":
    print("Hello!")
elif e == "B":
    print("Hi!")
```

The following code

```python
match e0, e1:
    case "A", 1:
        print("Hello!")
    case "B", 2:
        print("Hi!")
```

is rewritten as

```python
# match e0, e1:
#     case "A", 1:
#         print("Hello!")
#     case "B", 2:
#         print("Hi!")
if e0 == "A" and e1 == 1:
    print("Hello!")
elif e0 == "B" and e1 == 2:
    print("Hi!")
```

Anything more complex is currently not handled. Please fill an issue
if you need more cases to be covered.
