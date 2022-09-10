# Tokenizing text into tokens
While creating NLP models, our model expects numerical vectors rather than words because they cannot receive raw strings as input. The process of converting raw strings into encoded numerical vectors is known as **tokenization**. There are various ways of tokenizing the text mainly character tokenization and word tokenization.

## Character Tokenization
Consider a string, "I am learning NLP right now". While performing character tokenization, the given string is splitted into individual characters like 'I', 'a', 'm', and so on. This is a simple method of tokenization. We can perform character tokenization pythonically by converting string into list.

```python
text = "I am learning NLP right now"
tokenized_text = list(text)
print(tokenized_text)
```
```python
Output: ['I', ' ', 'a', 'm', ' ', 'l', 'e', 'a', 'r', 'n', 'i', 'n', 'g', ' ', 'N', 'L', 'P', ' ', 'r', 'i', 'g', 'h', 't', ' ', 'n', 'o', 'w']
```
