# Part of Speech Tagging
> This is my brief note from *DeepLearning.AI's* [NLP Specialization Course](https://www.coursera.org/specializations/natural-language-processing).

Part of Speech is the category of words or the the lexical terms in a language. Some of them are: 
  - noun
  - verb
  - adjective
  - pronoun
  - adverb and others

Consider a sentence: Why not learn something? This sentence can be represented in terms of part of speech where <code>why</code> can be categorized as <code>adverb</code>, <code>learn</code>as verb</code>, <code>something</code> as noun and so on.

During analysing large corpus of text, writing all these lexical terms may be cumbersome, so we write short form of these terms referring it as POS (Part of Speech).

## So, what is POS tagging?
Part of Speech tagging simply means assigning these tags to the word of a sentence or a corpus. POS tagging is applied in various areas of machine learning tasks such as:
  - Named entites
  - Co-reference resoultion
  - Speech recognition


## Markov Chains
Markov chains is a type of stochastic model that describes a sequence of possible events. It experiences transition from one state to another according to probabilistic rules. Using markov chain, to get probability of each events, it needs only the state of previous events.

Markov chains consist of a directed graph. The circles of the graph represents the state of the model. The change of state of the system are called transitions.
<p align="center">
  <img src="/images/markov_chains.png">
</p>
<div align="center"> Fig. Markov Chain State Diagram </div>

  
  ### Using Markov Chains for Part of Speech Tagging
  We will use transition probabilities introduced in Markov Chains to know the probabilities of going from one POS tag to another.
