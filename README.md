# Tobacco Consumption Clustering

## Project Description

This project analyzes tobacco consumption patterns across different states and regions in India using clustering techniques. The analysis is based on the Global Youth Tobacco Survey (GYTS) dataset, which contains comprehensive information about tobacco usage, exposure, awareness, and policies across various states and union territories in India.

### Purpose of the Research

The main objectives of this research are:

1. To identify patterns and similarities in tobacco consumption across different regions in India
2. To segment states/UTs based on various tobacco consumption indicators
3. To understand the relationship between different tobacco usage metrics
4. To provide data-driven insights that could inform targeted tobacco control policies

### Methodology

The project employs various data science and machine learning techniques:

1. **Data Preprocessing**: Cleaning the dataset, handling missing values, and scaling features
2. **Exploratory Data Analysis**: Visualizing distributions and relationships between variables
3. **Dimensionality Reduction**: Using Principal Component Analysis (PCA) to reduce the feature space
4. **Clustering**: Applying K-means clustering to identify natural groupings in the data
5. **Visualization**: Creating meaningful visualizations to interpret the clustering results

### Dataset

The analysis uses the GYTS4.csv dataset which contains tobacco consumption statistics across Indian states and union territories. Key metrics include:

- Tobacco usage rates (smoking and smokeless tobacco)
- Age of initiation
- Exposure to tobacco smoke
- Awareness about tobacco products
- Policy implementation indicators
- Tobacco cessation attempts

## Installation Instructions

### Prerequisites

- Python 3.7+
- pip (Python package installer)

### Setup Instructions for Windows

1. **Clone the repository**
   ```
   git clone https://github.com/yourusername/Tobacco-Consumption-Clustering.git
   cd Tobacco-Consumption-Clustering
   ```

2. **Create a virtual environment**
   ```
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install required packages**
   ```
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook**
   ```
   jupyter notebook
   ```

5. **Open the notebook**
   ```
   Tobacco_Consumption_Clustering.ipynb
   ```

### Setup Instructions for Linux

1. **Clone the repository**
   ```
   git clone https://github.com/yourusername/Tobacco-Consumption-Clustering.git
   cd Tobacco-Consumption-Clustering
   ```

2. **Create a virtual environment**
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install required packages**
   ```
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook**
   ```
   jupyter notebook
   ```

5. **Open the notebook**
   ```
   Tobacco_Consumption_Clustering.ipynb
   ```

## Key Findings

The clustering analysis reveals distinct patterns in tobacco consumption across different regions in India:

- Geographic variations in tobacco usage rates
- Correlation between tobacco policies and consumption rates
- Different patterns between urban and rural areas
- Relationship between age of initiation and current tobacco use

## Future Work

Potential extensions of this research include:
- Time-series analysis to track changes in tobacco consumption patterns
- Integration with additional socioeconomic and health indicators
- Development of predictive models for tobacco usage trends
- Comparative analysis with international tobacco consumption data

## License

This project is licensed under the MIT License - see the LICENSE file for details.
