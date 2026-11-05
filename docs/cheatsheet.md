# Cheat sheet: patterns used in this project

Add to this file whenever you look something up. Every example here was run and gives the output shown.
Quick question to Claude? Start the message with `Q:` for a one-to-three line answer.

## Python and pandas
```python
import pandas as pd
df = pd.read_csv("data/pairs_test.csv")            # load a table
df.shape                                           # (rows, columns)
df.columns.tolist()                                # column names
df.head(3)                                         # first 3 rows
df[df.label == "duplicate"]                        # rows where a condition is true
df[(df.label == "duplicate") & (df.name_a.str.len() > 20)]   # TWO conditions: use & or |, wrap each in ( )
df.label.value_counts()                            # count each value
pd.crosstab(df.label, df.is_duplicate)             # table of counts for two columns
for a, b in zip(df.name_a, df.name_b):             # loop over two columns together
    pass
```
**Remember:** `and` / `or` work on single True/False values. For whole columns or tensors use `&` / `|` with parentheses.

## Strings and fuzzy matching
```python
from rapidfuzz import fuzz
fuzz.token_sort_ratio("Science & Technology", "Science and Technology")   # 0-100, word order ignored
"univ" in "university of iowa"                      # True: substring test
"University of Iowa, Library".split(",")            # ['University of Iowa', ' Library']
```

## Tensors (PyTorch)
```python
import torch
x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
x.shape            # torch.Size([2, 3])  -> 2 rows, 3 columns
x.sum(dim=1)       # tensor([ 6., 15.])  -> collapse columns, one number per row
x.mean()           # tensor(3.5000)      -> one number
x * 2, x ** 2      # element by element, no loop needed
torch.clamp(x, min=2.0)                     # nothing below 2.0
mask = x > 3                                # True/False tensor, same shape as x
x[mask]            # tensor([4., 5., 6.]) -> keep only the True positions
x[mask].max()      # tensor(6.)
```

## Gradients
```python
w = torch.tensor([2.0, 3.0], requires_grad=True)   # PyTorch records everything done with w
loss = (w ** 2).sum()
loss.backward()                                     # fills w.grad with the slope of loss for each number
w.grad                                              # tensor([4., 6.])
# .item() or numpy() turns a tensor into a plain number and CUTS the record: do not use inside a loss
```

## A training loop, in words (you will write this one)
```
for each epoch:
    for each batch of examples:
        1. pred = model(batch)                  # forward pass
        2. loss = loss_function(pred, labels)   # how wrong is it
        3. optimizer.zero_grad()                # clear old gradients
        4. loss.backward()                      # compute new gradients
        5. optimizer.step()                     # nudge the weights against the gradient
```

## Reading an error
- Read the LAST line first: it names the error. The lines above show where.
- `shape mismatch` / `size must match`: print `.shape` of both tensors.
- `NameError`: a typo or a missing import. `KeyError`: a column or key that does not exist (check `df.columns`).
- `nan` in a result: usually a division by zero or a log of zero. Print the pieces.
