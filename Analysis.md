**Impact of Batch Size**
  1. A batch size of 128 provides a gradual and noticable decrease in loss compared to 64 or 256.

![SimCLR_Loss_BatchSize](https://github.com/user-attachments/assets/ec77ad7f-5f11-4a58-ac71-affcd6296eeb)

**Impact of LR**
  1. For a LARS Optimizer, the learning rate provided is base learning rate (LR). This is used as a starting point to calculate the learning rate for each layer based on trust ratio.
  2. Trust ratio compares the magnitude of the weights to the magnitude of the gradients. It essentially measures how much the optimizer can "trust" the gradient information for that layer.
  3. Based on the trust ratio, LARS adjusts the learning rate for each layer. If the trust ratio is high (meaning the gradient is reliable), the learning rate for that layer might be increased. Conversely, if the trust ratio is low (meaning the gradient is less reliable), the learning rate might be decreased.

![SimCLR_Loss_LR](https://github.com/user-attachments/assets/6f52da18-96f1-48fe-8f31-80b2f3b1e9c7)

**Impact of temperature coefficient**
  1. SimCLR loss function involves calculating similarity scores between augmented versions of the same image (positive pairs) and augmented versions of different images (negative pairs).
  2. Similarity scores are then passed through a softmax function to obtain probabilities.
  3. The temperature coefficient scales these similarity scores before they are passed to the softmax.
  4. Higher temperature: leads to softer probabilities, making the model less confident in distinguishing between similar and dissimilar pairs.
  5. Lower temperature: leads to sharper probabilities, making the model more sensitive to small differences between pairs

![SimCLR_Loss_Temp](https://github.com/user-attachments/assets/7c87fdac-5f36-4494-ae8f-e991a136f410)

