# 🔍 KNN Neighbors Visualizer

An interactive Streamlit application that allows you to visualize how the K-Nearest Neighbors (KNN) algorithm works by selecting points and seeing their nearest neighbors in real-time.

![KNN Visualizer](https://img.shields.io/badge/Interactive-Visualization-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)

## 📖 Overview

This application provides an intuitive way to understand the KNN algorithm by:

- **Visualizing** data points in 2D space using PCA
- **Interacting** with test points to see their k-nearest neighbors
- **Analyzing** how different parameters affect KNN predictions
- **Exploring** multiple datasets with different characteristics

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone or download this repository**
   ```bash
   git clone <repository-url>
   cd app
   ```

2. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser**
   - The app will automatically open in your default browser
   - If not, go to `http://localhost:8501`

## 📁 File Structure

```
app/
│
├── app.py          # Main Streamlit application 
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🎯 Features

### 📊 Multiple Datasets
- **Iris**: Classic classification dataset (3 classes, 4 features)
- **Wine**: Wine recognition dataset (3 classes, 13 features)
- **Breast Cancer**: Wisconsin breast cancer dataset (2 classes, 30 features)

### ⚙️ Customizable Parameters
- **k-value**: Number of neighbors (1-15)
- **Distance Metric**: Euclidean or Manhattan distance
- **Test Size**: Percentage of data for testing (10%-40%)
- **Dataset Selection**: Choose from 3 different datasets

### 👁️ Interactive Visualization
- **Point Selection**: Choose test points by index or random selection
- **Real-time Updates**: See immediate changes when parameters are adjusted
- **Hover Information**: Get details by hovering over points
- **Neighbor Lines**: Visual connections to k-nearest neighbors

### 📈 Comprehensive Analysis
- **Class Distribution**: Bar chart showing data balance
- **Neighbors Details**: Table with distances and coordinates
- **Voting Results**: Breakdown of how neighbors voted
- **Performance Metrics**: Overall model accuracy

## 🎮 How to Use

1. **Select Dataset**: Choose from Iris, Wine, or Breast Cancer in the sidebar
2. **Adjust Parameters**: 
   - Set the number of neighbors (k)
   - Choose distance metric
   - Adjust test set size
3. **Pick a Test Point**:
   - Select from dropdown or click "Pick Random"
   - See true class vs predicted class
4. **Analyze Results**:
   - View neighbors on the interactive plot
   - Check the neighbors details table
   - Examine voting results and model performance

## 🧠 Understanding KNN Through the App

### What You'll Learn
- How **k-value** affects decision boundaries
- Why **distance metrics** matter in different scenarios
- How **noise and outliers** impact predictions
- The importance of **feature scaling** in KNN

### Key Observations
- **Small k** (1-3): More complex boundaries, sensitive to noise
- **Large k** (7+): Smoother boundaries, more robust to noise
- **Euclidean**: Spherical decision boundaries
- **Manhattan**: Diamond-shaped decision boundaries

## 🛠️ Technical Details

### Algorithms Implemented
- **KNN Manual**: Custom implementation from scratch
- **PCA**: For 2D visualization of high-dimensional data
- **StandardScaler**: For feature normalization

### Libraries Used
- **Streamlit**: Web application framework
- **Plotly**: Interactive visualizations
- **Scikit-learn**: Machine learning utilities
- **Pandas**: Data manipulation
- **NumPy**: Numerical computations

## 📊 Sample Outputs

### Visualization Features
- **Training Points**: Colored by class with transparency
- **Test Points**: Gray diamonds
- **Selected Point**: Red star with dark border
- **Neighbors**: Green circles with black borders
- **Connection Lines**: Dashed orange lines to neighbors

### Information Display
- **True vs Predicted**: Immediate feedback on classification
- **Distance Measurements**: Exact distances to each neighbor
- **Vote Counting**: How each class voted in the prediction
- **Performance Stats**: Overall model accuracy

## 🔧 Customization

### Adding New Datasets
You can easily extend the app by modifying the `load_dataset` function:

```python
def load_dataset(dataset_name):
    if dataset_name == "Your Dataset":
        # Load your data here
        X, y = load_your_data()
        feature_names = ["feature1", "feature2", ...]
        target_names = ["class0", "class1", ...]
    # ... existing code ...
```

### Modifying Visualizations
The app uses Plotly for interactive plots. You can customize colors, markers, and layouts in the visualization section.

## ❓ Frequently Asked Questions

### Q: Why PCA for visualization?
A: PCA reduces high-dimensional data to 2D while preserving the most important variance, making it possible to visualize complex datasets.

### Q: What if the prediction is wrong?
A: Incorrect predictions often occur near decision boundaries or when neighbors are evenly split between classes. This is normal and educational!

### Q: Can I use my own data?
A: Yes! Modify the `load_dataset` function to load your custom CSV or dataset.

### Q: Why are features scaled?
A: KNN is distance-based, so features on different scales can dominate the distance calculation. Scaling ensures all features contribute equally.

## 🐛 Troubleshooting

### Common Issues
1. **Port already in use**: 
   ```bash
   streamlit run app.py --server.port 8502
   ```

2. **Package conflicts**:
   ```bash
   pip install --upgrade -r requirements.txt
   ```

3. **Plot not updating**: 
   - Check if you're using the correct Streamlit version
   - Try clearing browser cache

### Getting Help
- Check that all requirements are installed correctly
- Ensure you're using Python 3.8 or higher
- Try the simple version first: `streamlit run app.py`

## 📚 Learning Resources

### KNN Theory
- [Scikit-learn KNN Documentation](https://scikit-learn.org/stable/modules/neighbors.html)
- [Wikipedia: k-nearest neighbors algorithm](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)

### Streamlit
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Streamlit Gallery](https://streamlit.io/gallery)

### Data Visualization
- [Plotly Python Documentation](https://plotly.com/python/)
- [PCA Explanation](https://scikit-learn.org/stable/modules/decomposition.html#pca)

## 🤝 Contributing

Feel free to contribute to this project by:
- Adding new datasets
- Improving visualizations
- Enhancing user interface
- Fixing bugs

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Datasets provided by scikit-learn
- Built with Streamlit and Plotly
- Inspired by educational machine learning visualizations

---

**Happy Learning!** 🎓

Explore, experiment, and enjoy understanding KNN through interactive visualization!
```

## Additional Setup Instructions

### For Windows Users
```bash
# Create virtual environment (optional but recommended)
python -m venv knn_env
knn_env\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

### For Mac/Linux Users
```bash
# Create virtual environment (optional but recommended)
python3 -m venv knn_env
source knn_env/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

### For Deployment (Optional)
If you want to deploy this app online:

1. **Create account** on [Streamlit Community Cloud](https://streamlit.io/cloud)
2. **Connect your GitHub repository**
3. **Deploy automatically**

The deployment would require these additional files:

#### streamlit_app.py (for deployment)
```python
# Rename your main file to streamlit_app.py for easy deployment
# Or create this file that imports from your main file
from knn_visualizer import main

if __name__ == "__main__":
    main()
```

These files provide complete documentation and setup instructions for your KNN visualization application!