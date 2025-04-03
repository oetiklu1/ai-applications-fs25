# Run using [Lightning AI](https://lightning.ai/)

## Steps to Run:
1. **Create a Space** on Lightning AI.
2. **Upload Files:**  
   - [`requirements.txt`](requirements.txt)  
   - [`transfer_learning_image_classification.ipynb`](transfer_learning_image_classification.ipynb)
3. **Install Packages:**  
   Run the following command to install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. **Run the Notebook.**

## Notes:
- The second code block in the notebook handles **Hugging Face login**, which is required to upload the trained model. You need to generate an access token from [Hugging Face](https://huggingface.co/settings/tokens).
- It is **highly recommended** to use a **GPU instead of a CPU**, as training time is reduced by a factor of **24x** when using a GPU.


