# awesome-nlp-and-xai (Work in progress)

## Language Model ([Huggingface](https://huggingface.co/transformers/model_summary.html))

### Transformers ([Huggingface](https://huggingface.co/transformers/index.html))

#### Underlying Architecture

* Attention Is All You Need ([pdf](https://arxiv.org/pdf/1706.03762.pdf), [code](https://github.com/tensorflow/tensor2tensor), notes)

#### Encoder Only

* BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding ([pdf](https://arxiv.org/pdf/1810.04805.pdf), [code](https://github.com/google-research/bert), notes)
* RoBERTa: A Robustly Optimized BERT Pretraining Approach ([pdf](https://arxiv.org/pdf/1907.11692.pdf), [code](https://github.com/pytorch/fairseq/blob/master/examples/roberta/README.md), notes)
* ALBERT: A Lite BERT For Self-supervised Learning of Language Presentations ([pdf](https://arxiv.org/pdf/1909.11942.pdf), [code](https://github.com/google-research/ALBERT), notes)
* ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators ([pdf](https://arxiv.org/pdf/2003.10555.pdf), [code](https://github.com/google-research/electra), notes)
* XLM: Cross-lingual Language Model Pretraining ([pdf](https://arxiv.org/pdf/1901.07291.pdf), [code](https://github.com/facebookresearch/XLM), notes) 
* Longformer: The Long-Document Transformer ([pdf](https://arxiv.org/pdf/2004.05150.pdf), [code](https://github.com/allenai/longformer), notes)

#### Decoder Only

* GPT-1: Improving Language Understanding by Generative Pre-Training ([pdf](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf), [code](https://github.com/openai/finetune-transformer-lm), notes)
* GPT-2: Language Models are Unsupervised Multitask Learners ([pdf](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), [code](https://github.com/openai/gpt-2), notes)
* GPT-3: Language Models are Few-Shot Learners ([pdf](https://arxiv.org/pdf/2005.14165.pdf), [code](https://github.com/openai/gpt-3), notes)
* Reformer: The Efficient Transformer ([pdf](https://arxiv.org/pdf/2001.04451.pdf), [code](https://github.com/google/trax), notes)
* Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context ([pdf](https://arxiv.org/pdf/1901.02860.pdf), [code](https://github.com/kimiyoung/transformer-xl?utm_source=catalyzex.com), notes)
* XLNet: Generalized Autoregressive Pretraining for Language Understanding ([pdf](https://arxiv.org/pdf/1906.08237.pdf), [code](https://github.com/zihangdai/xlnet), notes)

#### Encoder + Decoder 

* PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization ([pdf](https://arxiv.org/pdf/1912.08777.pdf), [code](https://github.com/google-research/pegasus), notes)
* BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension ([pdf](https://arxiv.org/pdf/1910.13461.pdf), [code](https://github.com/pytorch/fairseq/blob/master/examples/bart/README.md), notes)
* T5: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer ([pdf](https://arxiv.org/pdf/1910.10683.pdf), [code](https://github.com/google-research/text-to-text-transfer-transformer), notes)
* ProphetNet: Predicting Future N-gram for Sequence-to-Sequence Pre-training ([pdf](https://arxiv.org/pdf/2001.04063.pdf), [code](https://github.com/microsoft/ProphetNet), notes) 
 
#### Multimodal

* Supervised Multimodal Bitransformers for Classifying Images and Text ([pdf](https://arxiv.org/pdf/1909.02950.pdf), [code](https://github.com/facebookresearch/mmbt), notes)

#### Retrieval-based

* DPR: Dense Passage Retrieval for Open-Domain Question Answering ([pdf](https://arxiv.org/pdf/2004.04906.pdf), [code](https://github.com/facebookresearch/DPR), notes)
* RAG: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks ([pdf](https://arxiv.org/pdf/2005.11401.pdf), [code](https://huggingface.co/transformers/model_doc/rag.html), notes)

#### Evaluation

* SentEval

#### Analysis

[comment]: ([pdf](), [code](), notes)

* Out of Order: How important is the sequential order of words in a sentence in Natural Language Understanding tasks? ([pdf](https://arxiv.org/pdf/2012.15180.pdf), code, notes)
* A Primer in BERTology: What we know about how BERT works ([pdf](https://arxiv.org/pdf/2002.12327.pdf), notes)
* Unnatural Language Inference ([pdf](https://arxiv.org/pdf/2101.00010.pdf), code, notes)
* oLMpics -- On what Language Model Pre-training Captures ([pdf](https://arxiv.org/pdf/1912.13283.pdf), [code](https://github.com/alontalmor/oLMpics), notes)
* Analyzing the Structure of Attention in a Transformer Language Model ([pdf](https://www.aclweb.org/anthology/W19-4808.pdf), [code](), notes)

#### Optimization

* Addressing Some Limitations of Transformers with Feedback Memory ([pdf](https://arxiv.org/pdf/2002.09402.pdf), notes)

#### Datasets / Benchmarks

* GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding ([pdf](https://arxiv.org/pdf/1804.07461.pdf), [code](https://github.com/nyu-mll/GLUE-baselines), notes)
* SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding SystemsCODE ([pdf](https://arxiv.org/pdf/1905.00537.pdf), [code](https://github.com/nyu-mll/jiant), notes)
* Adversarial NLI: A New Benchmark for Natural Language Understanding ([pdf](https://arxiv.org/pdf/1910.14599.pdf), [code](https://github.com/facebookresearch/anli), notes)
* The GEM Benchmark: Natural Language Generation, its Evaluation and Metrics [comment]: ([pdf](https://arxiv.org/pdf/2102.01672.pdf), [code](), notes)
* Did Aristotle Use a Laptop? A Question Answering Benchmark with Implicit Reasoning Strategies ([pdf](https://arxiv.org/pdf/2101.02235.pdf), [code](), notes)
* Long Range Arena: A Benchmark for Efficient Transformers ([pdf](https://arxiv.org/pdf/2011.04006.pdf), [code](https://github.com/google-research/long-range-arena), notes)

#### Survey

#### Wait for arrangement

* Switch Transformers: Scaling To Trillion Parameter Models With Simple and Efficient Sparsity ([pdf](https://arxiv.org/pdf/2101.03961.pdf), [code](https://github.com/lab-ml/nn/tree/master/labml_nn/transformers/switch), notes)
* Shortformer: Better Language Modeling using Shorter Inputs ([pdf](https://arxiv.org/pdf/2012.15832.pdf), [code](https://github.com/ofirpress/shortformer), notes)
* DeBERTa: Decoding-enhanced BERT with Disentangled Attention ([pdf](https://arxiv.org/pdf/2006.03654.pdf), [code](https://github.com/microsoft/DeBERTa), notes)
* Extracting Training Data from Large Language Models ([pdf](https://arxiv.org/pdf/2012.07805.pdf), notes)
 
SpanBERT
SentenceBERT
 
### Multitask Learning

### Transfer Learning

### Meta Learning

### Zero/Few-shot Learning

* Making Pre-trained Language Models Better Few-shot Learners ([pdf](https://arxiv.org/pdf/2012.15723v1.pdf), [code](https://github.com/princeton-nlp/LM-BFF), notes)

### Question Answering

* Studying Strategically: Learning to Mask for Closed-book QA ([pdf](https://arxiv.org/pdf/2012.15856.pdf), [code](), notes) 
* Multi-hop Question Answering via Reasoning Chains ([pdf](https://arxiv.org/pdf/1910.02610.pdf), [code](), notes) 

## Miscellaneous

MARS-Gym: A Gym framework to model, train, and evaluate Recommender Systems for Marketplaces ([pdf](https://arxiv.org/pdf/2010.07035v1.pdf), [code](https://github.com/deeplearningbrasil/mars-gym), notes)


[comment]: ([pdf](), [code](), notes)

## Multimodality

* Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet ([pdf](https://arxiv.org/pdf/2101.11986.pdf), [code](https://github.com/yitu-opensource/T2T-ViT), notes)
* CLIP: Learning Transferable Visual Models From Natural Language Supervision ([pdf](https://cdn.openai.com/papers/Learning_Transferable_Visual_Models_From_Natural_Language_Supervision.pdf), [code](https://github.com/openai/CLIP), notes)
* DALL-E ([blog](https://openai.com/blog/dall-e/))


## Interpretability

* A Survey on Neural Network Interpretability ([pdf](https://arxiv.org/pdf/2012.14261.pdf), [code](), notes)
* Interpretation of NLP models through input marginalization ([pdf](https://arxiv.org/pdf/2010.13984.pdf), notes)


