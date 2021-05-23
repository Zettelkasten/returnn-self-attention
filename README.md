# Self-Attention implementation for [RETURNN](https://github.com/rwth-i6/returnn)

![CI Badge](https://github.com/Zettelkasten/returnn-self-attention/actions/workflows/main.yml/badge.svg)

This repository implements self attention mechanisms:
 - "Vanilla" multi-head dot-attention of the [Transformer](https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf).
 - Locality sensitive hashing as presented in the [Reformer](https://arxiv.org/pdf/2001.04451.pdf).

Running this requires some features of [this RETURNN development branch](https://github.com/rwth-i6/returnn/tree/frithjof-self-attention),
e.g. the `CumConcatLayer` and being able to specify optional time axes by dim tag name (e.g. `stag:key-window?`).

Note that RETURNN also provides a `SelfAttentionLayer`,
which has the same effects as the vanilla self-attention implementation provided here
(there is also a test case checking this).
However, this implementation uses multiple layers to achieve the same effect.
This makes it extendable and clearer in what is actually going on.
