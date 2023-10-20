# Medical-VQA
Adapted from : [Open-Ended Medical Visual Question Answering Through Prefix Tuning of Language Models](https://arxiv.org/abs/2303.05977)<br>
`Dataset`: VQA-RAD<br>
`Visual Encoder`: https://huggingface.co/flaviagiammarino/pubmed-clip-vit-base-patch32<br>
`LM Decoder`: https://huggingface.co/cemilcelik/distilgpt2_pubmed<br>

The paper proposes open ended VQA training on large medical datasets by using encoded QA pair and image prefixes in the prompt to the LM decoder. Tried to adapt this to a much smaller dataset by using medical domain adaped pre-train models, didn't worked well.
