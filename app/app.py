import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter

class KNNManual:
    def __init__(self, k=3, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric
        
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self
        
    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)
    
    def _predict(self, x):
        # Calculate distances
        if self.distance_metric == 'euclidean':
            distances = [np.sqrt(np.sum((x - x_train)**2)) for x_train in self.X_train]
        elif self.distance_metric == 'manhattan':
            distances = [np.sum(np.abs(x - x_train)) for x_train in self.X_train]
        
        # Get k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        k_labels = [self.y_train[i] for i in k_indices]
        
        # Majority vote
        most_common = Counter(k_labels).most_common(1)
        return most_common[0][0]
    
    def get_neighbors(self, x, k=None):
        if k is None:
            k = self.k
            
        # Calculate distances
        if self.distance_metric == 'euclidean':
            distances = [np.sqrt(np.sum((x - x_train)**2)) for x_train in self.X_train]
        elif self.distance_metric == 'manhattan':
            distances = [np.sum(np.abs(x - x_train)) for x_train in self.X_train]
        
        # Get k nearest neighbors
        k_indices = np.argsort(distances)[:k]
        k_distances = [distances[i] for i in k_indices]
        k_points = [self.X_train[i] for i in k_indices]
        k_labels = [self.y_train[i] for i in k_indices]
        
        return k_indices, k_points, k_distances, k_labels

def load_dataset(dataset_name):
    """Load and prepare the selected dataset"""
    if dataset_name == "Iris":
        data = load_iris()
    elif dataset_name == "Wine":
        data = load_wine()
    elif dataset_name == "Breast Cancer":
        data = load_breast_cancer()
    
    X = data.data
    y = data.target
    feature_names = data.feature_names
    target_names = data.target_names
    
    return X, y, feature_names, target_names

def main():
    st.set_page_config(page_title="KNN Visualizer", page_icon="🔍", layout="wide")
    
    st.title("🔍 KNN Neighbors Visualizer")
    st.markdown("Select a point and visualize its k nearest neighbors interactively!")
    
    # Initialize session state for test point index
    if 'test_point_idx' not in st.session_state:
        st.session_state.test_point_idx = 0
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Dataset selection
    dataset_option = st.sidebar.selectbox(
        "Select Dataset",
        ["Iris", "Wine", "Breast Cancer"]
    )
    
    try:
        # Load data
        X, y, feature_names, target_names = load_dataset(dataset_option)
    except Exception as e:
        st.error(f"Error loading {dataset_option} dataset: {e}")
        st.stop()
    
    # K value selection
    k = st.sidebar.slider("Number of Neighbors (k)", 1, 15, 5)
    
    # Distance metric
    distance_metric = st.sidebar.selectbox(
        "Distance Metric",
        ["euclidean", "manhattan"]
    )
    
    # Test size
    test_size = st.sidebar.slider("Test Set Size (%)", 10, 40, 30) / 100
    
    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply PCA for 2D visualization
    pca = PCA(n_components=2)
    X_train_2d = pca.fit_transform(X_train_scaled)
    X_test_2d = pca.transform(X_test_scaled)
    
    # Train KNN model
    knn = KNNManual(k=k, distance_metric=distance_metric)
    knn.fit(X_train_scaled, y_train)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Dataset Information")
        st.write(f"**Dataset**: {dataset_option}")
        st.write(f"**Samples**: {X.shape[0]}")
        st.write(f"**Features**: {X.shape[1]}")
        st.write(f"**Classes**: {len(target_names)}")
        st.write(f"**Training samples**: {len(X_train)}")
        st.write(f"**Test samples**: {len(X_test)}")
        
        # Class distribution
        st.subheader("📈 Class Distribution")
        class_counts = pd.Series(y).value_counts().sort_index()
        fig_bar = px.bar(
            x=[target_names[i] for i in class_counts.index],
            y=class_counts.values,
            labels={'x': 'Class', 'y': 'Count'},
            title="Class Distribution"
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Select Test Point")
        
        # Test point selection
        test_point_idx = st.selectbox(
            "Select test point by index:",
            options=list(range(len(X_test))),
            index=st.session_state.test_point_idx,
            format_func=lambda x: f"Point {x} (True class: {target_names[y_test[x]]})"
        )
        
        # Update session state when user selects a point
        st.session_state.test_point_idx = test_point_idx
        
        # Or random selection
        if st.button("🎲 Pick Random Test Point"):
            st.session_state.test_point_idx = np.random.randint(0, len(X_test))
            st.rerun()
        
        # Display test point info
        test_point_original = X_test[st.session_state.test_point_idx]
        test_point_scaled = X_test_scaled[st.session_state.test_point_idx]
        test_point_2d = X_test_2d[st.session_state.test_point_idx]
        true_label = y_test[st.session_state.test_point_idx]
        
        st.write(f"**True class**: {target_names[true_label]}")
        st.write(f"**Coordinates (2D)**: ({test_point_2d[0]:.3f}, {test_point_2d[1]:.3f})")
        
        # Make prediction
        prediction = knn.predict([test_point_scaled])[0]
        st.write(f"**Predicted class**: {target_names[prediction]}")
        
        if prediction == true_label:
            st.success("✅ Correct prediction!")
        else:
            st.error("❌ Incorrect prediction!")
    
    # Visualization
    st.subheader("👁️ Neighbors Visualization")
    
    # Get neighbors information
    neighbor_indices, neighbor_points, neighbor_distances, neighbor_labels = knn.get_neighbors(
        test_point_scaled, k=k
    )
    
    # Convert neighbors to 2D for visualization
    neighbor_points_2d = pca.transform(neighbor_points)
    
    # Create interactive plot
    fig = go.Figure()
    
    # Plot training points
    for class_idx in range(len(target_names)):
        class_mask = y_train == class_idx
        fig.add_trace(go.Scatter(
            x=X_train_2d[class_mask, 0],
            y=X_train_2d[class_mask, 1],
            mode='markers',
            marker=dict(size=8, opacity=0.6),
            name=f"Train: {target_names[class_idx]}",
            hovertemplate=f"Class: {target_names[class_idx]}<br>X: %{{x:.3f}}<br>Y: %{{y:.3f}}<extra></extra>"
        ))
    
    # Plot test points (excluding selected one)
    other_test_mask = np.ones(len(X_test_2d), dtype=bool)
    other_test_mask[st.session_state.test_point_idx] = False
    fig.add_trace(go.Scatter(
        x=X_test_2d[other_test_mask, 0],
        y=X_test_2d[other_test_mask, 1],
        mode='markers',
        marker=dict(size=10, color='gray', symbol='diamond', opacity=0.7),
        name="Other test points",
        hovertemplate="Test point<br>X: %{{x:.3f}}<br>Y: %{{y:.3f}}<extra></extra>"
    ))
    
    # Plot selected test point
    fig.add_trace(go.Scatter(
        x=[test_point_2d[0]],
        y=[test_point_2d[1]],
        mode='markers',
        marker=dict(size=20, color='red', symbol='star', line=dict(width=2, color='darkred')),
        name=f"Selected: {target_names[true_label]}",
        hovertemplate=f"Selected point<br>True: {target_names[true_label]}<br>X: %{{x:.3f}}<br>Y: %{{y:.3f}}<extra></extra>"
    ))
    
    # Plot neighbors - group them by class to reduce legend items
    neighbor_groups = {}
    for i, (point_2d, label, distance) in enumerate(zip(neighbor_points_2d, neighbor_labels, neighbor_distances)):
        if label not in neighbor_groups:
            neighbor_groups[label] = {
                'x': [], 'y': [], 'distances': [], 'indices': []
            }
        neighbor_groups[label]['x'].append(point_2d[0])
        neighbor_groups[label]['y'].append(point_2d[1])
        neighbor_groups[label]['distances'].append(distance)
        neighbor_groups[label]['indices'].append(i+1)
    
    # Add one trace per class for neighbors
    for label, data in neighbor_groups.items():
        # Custom hover text for each point in the group
        hover_text = []
        for i, (x, y, dist, idx) in enumerate(zip(data['x'], data['y'], data['distances'], data['indices'])):
            hover_text.append(f"Neighbor {idx}<br>Class: {target_names[label]}<br>Distance: {dist:.3f}<br>X: {x:.3f}<br>Y: {y:.3f}")
        
        fig.add_trace(go.Scatter(
            x=data['x'],
            y=data['y'],
            mode='markers',
            marker=dict(size=15, color='lime', symbol='circle', line=dict(width=2, color='green')),
            name=f"Neighbors: {target_names[label]}",
            hovertemplate="%{text}<extra></extra>",
            text=hover_text
        ))
        
        # Draw lines to neighbors
        for x, y in zip(data['x'], data['y']):
            fig.add_trace(go.Scatter(
                x=[test_point_2d[0], x],
                y=[test_point_2d[1], y],
                mode='lines',
                line=dict(color='orange', width=2, dash='dash'),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # Update layout with better legend configuration
    fig.update_layout(
        title=f"KNN Visualization (k={k}) - {dataset_option} Dataset",
        xaxis_title=f"Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)",
        yaxis_title=f"Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)",
        width=800,
        height=600,
        legend=dict(
            orientation="v",  # Vertical orientation
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='Black',
            borderwidth=1,
            font=dict(size=10)  # Smaller font
        ),
        margin=dict(l=50, r=150, t=50, b=50)  # Add right margin for legend
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Neighbors details table
    st.subheader("📋 Neighbors Details")
    
    neighbors_data = []
    for i, (idx, label, distance) in enumerate(zip(neighbor_indices, neighbor_labels, neighbor_distances)):
        neighbors_data.append({
            "Neighbor": i+1,
            "Training Index": idx,
            "Class": target_names[label],
            "Distance": f"{distance:.4f}",
            "Coordinates (2D)": f"({neighbor_points_2d[i][0]:.3f}, {neighbor_points_2d[i][1]:.3f})"
        })
    
    neighbors_df = pd.DataFrame(neighbors_data)
    st.dataframe(neighbors_df, use_container_width=True)
    
    # Voting results
    st.subheader("🗳️ Voting Results")
    vote_counts = Counter(neighbor_labels)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Neighbors", k)
    
    with col2:
        winning_class = max(vote_counts.items(), key=lambda x: x[1])[0]
        st.metric("Winning Class", target_names[winning_class])
    
    with col3:
        st.metric("Winning Votes", vote_counts[winning_class])
    
    # Vote breakdown
    st.write("**Vote Breakdown:**")
    for class_idx, count in vote_counts.items():
        st.write(f"- {target_names[class_idx]}: {count} vote(s)")

if __name__ == "__main__":
    main()