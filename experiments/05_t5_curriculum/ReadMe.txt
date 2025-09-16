Explanation of codes in the folders:
Part1:
First code: t5-small model trained on babylm from scratch with random weights with no vector transition (VT).
other codes: the same model with vector transition of first, middle and last layers with different weights.

Part2:
First code: training t5-small model from scratch with curriculum-only approach without VT.
second code: training the same model from scratch with curriculum-only approach with VT of first layer.

Part3: 
First code: finetuning the t5-small model with pretrained weights on babylm dataset without curriculum learning.
Second code: finetuning t5-small model with pretrained weight on the same dataset with curriculum-only approach.

Part4:
First code: finetuning t5-small model on babylm dataset with different masking level approach

Part5:
First code: training t5-small model from scratch with random weights on babylm dataset with hybrid-curriculum approach without VT and without penalty concepts.


Part6:
First code: training t5-small model from scratch with random weights on babylm dataset with hybrid-curriculum approach with VT and without penalty concepts.

evaluation code: 
This is the code we used to evaluate our models on evaluation datasets that enables us to compare the performance of the model. the evaluation datasets are not used dutring training.
