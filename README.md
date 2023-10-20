# Medical-VQA
Adapted from : [Open-Ended Medical Visual Question Answering Through Prefix Tuning of Language Models](https://arxiv.org/abs/2303.05977)<br>
`Dataset`: VQA-RAD<br>
`Visual Encoder`: https://huggingface.co/flaviagiammarino/pubmed-clip-vit-base-patch32<br>
`LM Decoder`: https://huggingface.co/cemilcelik/distilgpt2_pubmed<br>

Tried to adapt a domain specific training approach on large datsets proposed in the paper for finetuning a small medical datatset by using medical domain adapted pre-trained models. Didn't worked well. 
