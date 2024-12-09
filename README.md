# Classifier-Free Guidance Diffusion

This repository contains the **Classifier-Free Guidance Diffusion** project, developed as part of the **Generative Models** course taught by Alain Durmus at **École Polytechnique**. This project explores and implements concepts from the paper ["Classifier-Free Diffusion Guidance"](https://arxiv.org/abs/2101.04775), which introduces innovative methods to guide diffusion models effectively during sampling.

---

## 📚 Project Overview
The primary objective of this project is to deeply understand the *Classifier-Free Guidance* mechanism to simplify diffusion pipelines and improve the quality of generated results. This approach guides diffusion models without requiring an external classifier, making the process more flexible and efficient.

---

## ⚙️ Project Content
1. **Paper Review**:
   - Summary and explanation of key concepts.
   - Analysis of the theoretical implications for generative modeling.

2. **Implementation**:
   - Development of a guidance mechanism for diffusion models without a classifier.
   - Training and evaluation scripts for the model.

3. **Experiments**:
   - Tests to evaluate the quality of samples generated using the guidance mechanism.
   - Comparison with other diffusion guidance techniques.

4. **Results**:
   - Visualizations and metrics demonstrating the impact of *Classifier-Free Guidance*.
   - Discussion of the advantages and limitations of this approach.

---


## 🚀 Installation and Usage

### Requirements
- Python 3.8+
- PyTorch
- Additional dependencies listed in `requirements.txt`

### Installation
```bash
git clone https://github.com/<your-username>/classifier-free-guidance-diffusion.git
cd classifier-free-guidance-diffusion
pip install -r requirements.txt
```


### Running the Code

- **To train the model**, use the command:
  ```bash
  python train.py
  ```


- **To generate samples with guidance**, use the command:
  ```bash
  python sample.py --guidance-scale <value>
  ```


---

## 📊 Key Results
- Improved sample quality compared to baseline diffusion methods.
- Reduced reliance on external classifiers, simplifying the training pipeline.

---

## 📖 References
- Original paper: *Classifier-Free Diffusion Guidance* [Ho et al., 2021](https://arxiv.org/abs/2101.04775).
- Relevant frameworks and libraries: PyTorch, Hugging Face Diffusers, etc.

---

## 🙌 Acknowledgments
Special thanks to Alain Durmus for his guidance throughout the **Generative Models** course, and to École Polytechnique for providing this opportunity to explore cutting-edge AI research.

Feel free to explore this repository and reach out for any questions or collaborations! 🚀

