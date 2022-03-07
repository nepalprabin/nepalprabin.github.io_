# Autocorrect and Minimum Edit Distance
> This is my brief note from *DeepLearning.AI's* [NLP Specialization Course](https://www.coursera.org/specializations/natural-language-processing).

## What is Autocorrect?
Autocorrect is an application that changes misspelled word into a correct word. When writing messages or drafting an email, you may have noticied that if we type any words
that is misspelled, then that word automatically gets corrected with correct spelling and based on the context.

<p align="center">
  <img src="/images/autocorrect_example.png">
</p>
<div align="center"> Fig. Autocorrect in action in google document </div>

## How does autocorrect work??
While typing the document we can see we get automatic correction in our document. The basic working of this automatic correction is:
  - Identifying a misspelled word
  - Find the strings <code>n</code> edit distance away
  - Filter the candidates
  - Calculate word probabilities

Now let's see each of the points in detail.

#### 1. Identifying a misspelled word:
  Let's say we are writing a sentence <code>This is a draft docment of the APIs</code>. Here we can see clearly see that the word <code>docment</code> is misspelled.
  But, how do we know that this is a misspelled word? Well, we will have a dictionary containing all correct words and if we do not encounter given string in the dictionary, 
  that string is obviously a misspelled word.
  ``` python
  if word not in vocab:
    misspelled = True       # If the word is not in vocab, we identify it as a misspelled word. 
  ```
  While identifying a misspelled words, we are only looking at the vocab but not the context. Consider a sentence <code>Hello deah</code>. Here, <code>dear</code> is misspelled
  as <code>deah</code>. If we write <code>deer</code> instead of <code>dear</code>, then we would not be able to identify misspelled word because <code>deer</code> is present in vocab,
  though it is contextually incorrect.
  
#### 2. Find strings n edit distance away
  Edit is an operation that is performed on a string to change it. 
  - Types of edit:
      - Insert          (add a letter)                  <code>'to': 'top', 'two'</code>
      - Delete          (remove a letter)               <code>'hat': 'ha', 'at', 'ht'</code>
      - Switch          (swap 2 adjacent letters)       <code>'eta': 'eat', 'tea'</code>
      - Replace         (change 1 letter to another)    <code>'jaw': 'jar', 'paw'</code>
Using these edits, we can find all possible strings that are <code>n</code> edits away. 

#### 3. Filter candidates
  After findings strings that are n edit distance away, next step is to filter those strings. After applying edits, the strings are compared with the vocab, and if 
  those strings are not present in vocab, they are discarded. This, way we get a list of actual words.
  
#### 4. Calculate the word probabilities
  The final step is to calculate the word probabilities and find the most likely word from the vocab. Given the sentence <code>I am learning AI because AI is the new 
  electricity</code>, we find occurrence of each word and calculate the probability. Probability of given word <code>w</code> can be calculated as the ratio of the count
  of word <code>w</code> to the total size of the corpus.
  Mathematically:
  $P(w) = \frac{C(w)}{V}$,
  
  where:
  - $P(w) - Probability \ of\ a\ word$
  - $C(w) - Number \ of \ times \ the \ word \ appears$
  - $V    - Total \ size \ of \ the \ corpus$
  

## Minimum Edit Distance
  Till now, we have seen how edit distance works and what are its applications in NLP domain. Now let's look at the <i>minimum edit distance</i>.
  
  <b>Minimum edit distance</b> is the minimum number of edits required to transform one string to another. It is used in various applications such as spelling correction, machine translation, DNA sequencing, and many more. To calculate minimum edit distance, we use three types of operations which is also discussed above, i.e., insert, delete and replace.
  
  For example:
  Consider source word <code>deer</code> and target word <code>door</code>. To change source to target, we need to perform two replace operations i.e., replace each of <code>e</code> to <code>o</code>. Here number of edits is 2, but in minimum distance distance, there are cost associated with different edit operations.

  |   Edit Operations        | Cost    |
  | -----------|:----:|
  | Insert     | 1    |
  | Delete     | 1    |
  | Replace    | 2    |

Using above table, the edit distance for the above problem is: Edit distance = <code>2 x replace cost</code> = <code>2x2</code> = <code>4</code>

This is a brute force method where we are simply looking at the source and target word to calculate the edit distance. If we have large sentence, calculating edit distance with mentioned approach becomes tedious. To make things simpler, we can opt for tabular method using Dynamic Programming approaches.
   

