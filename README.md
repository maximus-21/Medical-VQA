# Medical-VQA

## Dataset 

The dataset used here is [**VQA-RAD**](https://huggingface.co/datasets/flaviagiammarino/vqa-rad). It is a dataset of clinically generated visual questions and answers about radiology images. The dataset consists of two types of questions: CLOSED and OPEN.<br>

Closed standard questions are with one word answers where the answer is either “yes”, “no” or one of the options mentioned in the question itself. Examples:<br>
1. Is the patient male or female ? <br>
2. Is the lesion on the left or the right side of the brain ?<br>
3. Is this an mri ?<br>

Open standard questions on the other have open ended answers. The answer could be more  than one word also, possibly a phrase.  Examples:<br>
1. Size of the mass in the upper right quadrant ?<br>
2. How was this image taken ?<br>
3. What is wrong with the ventricles ?<br>

The subject of the questions are  position, modality, plane, abnormality , attributes etc. <br>

## Approaches 

During the literature survey we found there are two methods for the Visual Question Answering: Classification and Generation.<br>

**Classification**: The majority of methods are classification-based and make use of different types of encoders, such as CNNs or Transformers followed by a classification layer. <br>

**Generation**: It involves encoding image features and textual embeddings, fusing them together and feeding into a LM Decoder for generating the answers. <br> 

Since VQA-RAD consisted of both closed and open ended questions, we tried the generation approaches. <br>

## Generation Model

[**PMC-VQA**](https://arxiv.org/abs/2305.10415) is one of the best-performing models for the medical VQA. The paper proposes two methods: encoder-based and decoder-based models named **MedVInT-TE** and **MedVInT-TD**, respectively. MedVInT-TE treats the problem similarly to the classification method whereas, MedVInT-TD consists of an LM decoder and is similar to the generation method. <br>

We tried to adopt the methodology from **MedVInT-TD** for both open ended and closed form QA pairs as our baseline. <br>

### Baseline Methodology : 

1. The images are fed into a visual decoder to get image features and the question embeddings are generated using the same pre-trained model used as the LM decoder for generating the answers. 
2. The extracted image features are mapped to the same embedding length as that of the question embeddings using feed forward layers to match their latent space.
3. Both the embeddings are concatenated and fed into the decoder for generating the answers.

### Training:

1. Considering the size of the dataset, It wasn’t possible to use pre-trained models to adapt to the medical domain using VQA-RAD so, instead we used versions of the pre-trained model that were already adapted to the medical domain through training on large medical datasets.
2. Keeping in mind the compute resources we chose:
     1. **Visual Encoder** : [**PubMed CLIP VIT Base P32**](https://huggingface.co/flaviagiammarino/pubmed-clip-vit-base-patch32)
      2. **LM Decoder** : [**Distill PubMed GPT2**](https://huggingface.co/cemilcelik/distilgpt2_pubmed)
3. We finetuned the model on the VQA-RAD by training similarly to the language modeling task using the logits with binary cross entropy loss for each generated token in the PyTorch Framework.
4. The model generated closed forms such as “yes”, “no”, “right”, “axial” etc but didn’t perform well with open ended questions.
5. Also in the baseline we face the problem of repetition generation within the answer, probably since we didn’t use any decoding method here such as Beam Search.


Choice for PubMed CLIP was made after the following paper: [**Does CLIP Benefit Visual Question Answering in the Medical Domain as Much as it Does in the General Domain?**](https://arxiv.org/abs/2112.13906) <br>

[**Notebook for the baseline**](https://www.kaggle.com/code/aaaacash/open-ended-vqa/notebook)

### Baseline Improvements:

During the literature review we came across an exciting paper : [**Open Ended Medical VQA Through Prefix Tuning of LMs**](https://arxiv.org/abs/2303.05977). <br>

Instead of concatenating the text and image embedding and feeding it to the LM decoder the idea is to use the question and answer and [**image prefixes**](https://arxiv.org/abs/2111.09734) as a prompt to the LM decoder. <br>

#### Prompt Template : 

p = [ **question**: What does the right side of the field show?   **context**: _v1, v2, . . . vx_    **answer**:  ]<br>

Here, p is the prompt and [_v1, v2,.....vx_] are the image prefixes generated through visual decoder. <br>

![Screenshot from 2023-10-21 21-46-22](https://github.com/maximus-21/Medical-VQA/assets/98597396/31f1d21e-dd5a-4038-9b51-7d9f591529c3)


#### Training and Results :

1. The visual decoder, LM decode and mapper remain the same as in the baseline. 
2. For the training purpose the answer tokens are added to the prompt template and during the evaluation the answer tokens remain empty as shown in the prompt template p.
3. The value of x i.e. no of image prefixes we took is 1. In the paper it was 2. 
4. Training is done similar to the baseline with the logits using BCE loss for 10 epochs with the batch size of 16 and learning rate of 5e-3 (as in the paper). 
5. For the evaluation we adopted the beam decoding method.
6. There were improvements to the baseline, we didn’t face any word repetitions problem during the decoding.
7. The model, although perfectly learned to generate domain related sensible answers, didn't perform well and often generated wrong answers. We achieved accuracy of 0.45 on Yes/No labels and it wasn’t possible to exactly get the accuracy for open ended answers.
8. The possible reason for this may be that unsupervised generative training doesn’t have any fixed answer space as in the case of classification, so the model might require a relatively large dataset to be properly finetuned. 
9. _Training with LoRA_ : We tried integrating LoRA to the GPT decoder using the PEFT library for parameter efficiency but it resulted in uneven training with loss decreasing halfway and then suddenly increasing. We then deprecated it. 

### Possible Third Approach

The VQA-RAD dataset was generated using the [**MedPix**](https://medpix.nlm.nih.gov/home) database where medical reports are present. In the public VQA dataset, there is a report link for each radiology image. A possible approach could be gathering the reports and building documents or vector databases and using a retrieval base model for question answering. Can fine-tune the retrieval model or could use [**Retrieval Augmented Generation (RAG)**](https://www.pinecone.io/learn/retrieval-augmented-generation/) for extracting the relevant context and use it in the prompt with the questions. It would be interesting to adopt RAG for multimodal use cases. Initially planned to use this with MIMIC-CXR Dataset until it was changed. 
