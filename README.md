# 📜 19th-Century Intellectual Network Explorer

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-239120?style=for-the-badge&logo=plotly&logoColor=white)

An interactive, web-based dashboard built with Streamlit that explores a synthetic dataset of 19th-century European intellectuals. This project visualizes complex relationships using dimensionality reduction (PCA, t-SNE), hierarchical clustering, and network graph theory.

## 🧠 The Premise

This application generates a fictional dataset of 120 intellectuals split evenly between two cities: **Paris** and **London**. Each intellectual belongs to one of three disciplines: *Literature*, *Natural Philosophy*, or *Political Economy*. 

The dataset is generated with specific humanistic constraints to simulate historical intellectual dynamics:
*   **The Paris Salon:** Intellectuals in Paris mingle across disciplines. A poet is just as likely to exchange letters with a political economist as they are with another poet.
*   **The London Silos:** Intellectuals in London are strictly siloed. Natural philosophers only speak to natural philosophers, simulating rigid academic boundaries.
*   **Feature Correlation:** Variables like `Avg_Sentence_Length` and `Sentiment_Polarity` are strongly correlated with the intellectual's discipline. 

## ✨ Features

*   **Zero-Setup Data:** The dataset is synthetically generated and cached in memory the moment the app loads. No CSV uploads are required.
*   **Interactive Network Graph:** Visualize the geographical and disciplinary clusters using `NetworkX` and `Plotly`. Hover over nodes to see character profiles.
*   **Dimensionality Reduction:** Project high-dimensional linguistic and wealth features into 2D space using **PCA** (Principal Component Analysis) or **t-SNE**.
*   **Hierarchical Clustering:** View mathematical relationships between intellectuals via an interactive **Dendrogram**.
*   **Dynamic Styling:** Instantly recolor visualizations based on categorical variables (`City` or `Discipline`).

## 🚀 Live Demo

*(Note: Add your Streamlit Community Cloud URL here once deployed)*
[**Launch the Web App**](https://your-app-url.streamlit.app)

## 💻 Running it Locally

If you want to run this application on your own machine, follow these steps:

**1. Clone the repository:**
```bash
git clone [https://github.com/yourusername/intellectual-network-app.git](https://github.com/yourusername/intellectual-network-app.git)
cd intellectual-network-app
