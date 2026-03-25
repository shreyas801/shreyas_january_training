"""
Flask API Server for Customer Segmentation Dashboard
=====================================================
This module provides API endpoints to serve customer segmentation
results to the frontend dashboard.

Author: Student
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from flask import Flask, jsonify, send_from_directory, render_template
from flask_cors import CORS

# Import custom modules
from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.clustering_models import ClusteringModels
from src.cluster_analysis import ClusterAnalyzer
from src.business_insights import BusinessInsightsGenerator

# Suppress warnings
warnings.filterwarnings('ignore')

# Initialize Flask app
app = Flask(__name__, static_folder='frontend', template_folder='frontend')
CORS(app)

# Global variable to store pipeline results
pipeline_results = None


def run_pipeline():
    """
    Run the complete customer segmentation pipeline and cache results.
    """
    global pipeline_results
    
    print("Running Customer Segmentation Pipeline...")
    
    # Initialize pipeline
    from main import CustomerSegmentationPipeline
    pipeline = CustomerSegmentationPipeline()
    
    # Load and process data
    pipeline.load_data()
    pipeline.preprocess_data()
    pipeline.engineer_features()
    
    # Train clustering models
    pipeline.train_clustering_models()
    
    # Perform cluster analysis
    cluster_profiles = pipeline.perform_cluster_analysis()
    
    # Generate business insights
    insights = pipeline.generate_business_insights(cluster_profiles)
    
    # Store results
    pipeline_results = {
        'pipeline': pipeline,
        'cluster_profiles': cluster_profiles,
        'insights': insights
    }
    
    return pipeline_results


def prepare_api_data():
    """
    Prepare data in format required by frontend API.
    """
    if pipeline_results is None:
        run_pipeline()
    
    pipeline = pipeline_results['pipeline']
    insights = pipeline_results['insights']
    cluster_profiles = pipeline_results['cluster_profiles']
    
    # Get cluster labels
    best_results = pipeline.cluster_results.get(pipeline.best_model, pipeline.cluster_results['kmeans'])
    labels = best_results['labels']
    
    # Prepare data with cluster labels
    data = pipeline.feature_engineered_data.copy()
    data['Cluster'] = labels
    
    # Calculate segment statistics
    segments = []
    segment_colors = ['#667eea', '#f093fb', '#43e97b', '#fa709a', '#4facfe']
    
    for cluster_id in sorted(data['Cluster'].unique()):
        cluster_data = data[data['Cluster'] == cluster_id]
        
        # Determine customer type based on metrics
        avg_spending = cluster_data['Total_Spending'].mean()
        avg_purchases = cluster_data['Num_Purchases'].mean()
        avg_recency = cluster_data['Recency'].mean()
        
        if avg_recency > 70:
            customer_type = 'At-Risk Customers'
            priority = 'URGENT'
            churn_risk = 'High'
        elif avg_spending > 3500:
            customer_type = 'Premium Loyal Customers'
            priority = 'HIGH'
            churn_risk = 'Low'
        elif avg_spending > 2500:
            customer_type = 'Big Ticket Buyers'
            priority = 'HIGH'
            churn_risk = 'Medium'
        elif avg_purchases > 25:
            customer_type = 'Frequent Low-Spenders'
            priority = 'MEDIUM'
            churn_risk = 'Low'
        else:
            customer_type = 'Budget Conscious Shoppers'
            priority = 'LOW'
            churn_risk = 'Medium'
        
        segments.append({
            'id': int(cluster_id),
            'name': f'Cluster {cluster_id}',
            'customerType': customer_type,
            'description': f'Cluster with avg spending ${avg_spending:,.0f}',
            'count': int(len(cluster_data)),
            'percentage': round((len(cluster_data) / len(data)) * 100, 1),
            'avgSpending': round(avg_spending, 2),
            'avgPurchases': round(avg_purchases, 1),
            'avgIncome': round(cluster_data['Income'].mean(), 2) if 'Income' in cluster_data.columns else 0,
            'avgAge': round(cluster_data['Age'].mean(), 1) if 'Age' in cluster_data.columns else 0,
            'avgSpendingScore': round(cluster_data['Spending_Score'].mean(), 1) if 'Spending_Score' in cluster_data.columns else 0,
            'avgRecency': round(avg_recency, 1),
            'priority': priority,
            'churnRisk': churn_risk,
            'revenue': round(cluster_data['Total_Spending'].sum(), 2),
            'revenuePercentage': round((cluster_data['Total_Spending'].sum() / data['Total_Spending'].sum()) * 100, 1),
            'color': segment_colors[cluster_id] if cluster_id < len(segment_colors) else '#667eea'
        })
    
    # Model performance
    model_performance = {}
    for model_name, results in pipeline.cluster_results.items():
        model_performance[model_name] = {
            'name': model_name.title() + ' Clustering',
            'silhouetteScore': round(results.get('silhouette', 0), 2),
            'daviesBouldin': round(results.get('davies_bouldin', 0), 2),
            'clusters': results.get('n_clusters', 0),
            'isBest': model_name == pipeline.best_model
        }
    
    # Prepare insights
    api_insights = {
        'revenueAnalysis': insights.get('revenue_analysis', {}),
        'churnAnalysis': insights.get('churn_analysis', {}),
        'keyFindings': insights.get('summary', {}).get('key_findings', []),
        'actionItems': insights.get('summary', {}).get('action_items', [])
    }
    
    # Prepare marketing strategies
    marketing_strategies = []
    for segment in segments:
        if 'Premium' in segment['customerType']:
            strategy = {
                'segment': segment['name'],
                'priority': 'HIGH',
                'approach': 'VIP Treatment & Exclusive Offers',
                'channels': ['Personal Email', 'Phone', 'VIP Events'],
                'offers': ['Early access to new products', 'Exclusive discounts', 'Personalized recommendations', 'Loyalty rewards program'],
                'budget': '35%',
                'color': segment['color']
            }
        elif 'At-Risk' in segment['customerType']:
            strategy = {
                'segment': segment['name'],
                'priority': 'URGENT',
                'approach': 'Win-back Campaigns',
                'channels': ['Email', 'SMS', 'Retargeting Ads'],
                'offers': ['Special comeback discounts', 'Limited-time offers', 'Feedback surveys', 'Personalized re-engagement'],
                'budget': '25%',
                'color': segment['color']
            }
        elif 'Big Ticket' in segment['customerType']:
            strategy = {
                'segment': segment['name'],
                'priority': 'HIGH',
                'approach': 'Premium Product Focus',
                'channels': ['Personal Email', 'Direct Mail', 'Showroom'],
                'offers': ['Premium product bundles', 'Financing options', 'White-glove service', 'Exclusive previews'],
                'budget': '25%',
                'color': segment['color']
            }
        elif 'Frequent' in segment['customerType']:
            strategy = {
                'segment': segment['name'],
                'priority': 'MEDIUM',
                'approach': 'Engagement & Upselling',
                'channels': ['Email', 'App Notifications', 'Social Media'],
                'offers': ['Bundle deals', 'Cross-sell recommendations', 'Loyalty points', 'Referral programs'],
                'budget': '20%',
                'color': segment['color']
            }
        else:
            strategy = {
                'segment': segment['name'],
                'priority': 'LOW',
                'approach': 'Value-Focused Marketing',
                'channels': ['Email', 'Social Media', 'Deals Pages'],
                'offers': ['Discount promotions', 'Clearance sales', 'Value bundles', 'Cashback offers'],
                'budget': '10%',
                'color': segment['color']
            }
        marketing_strategies.append(strategy)
    
    # Distribution data
    distribution = {
        'age': {
            'labels': ['18-25', '26-35', '36-45', '46-55', '56-65', '65+'],
            'data': [1200, 2500, 2800, 1900, 1100, 600]
        },
        'income': {
            'labels': ['<30K', '30-50K', '50-75K', '75-100K', '100-150K', '150K+'],
            'data': [1500, 2200, 2500, 1800, 1200, 800]
        },
        'purchases': {
            'labels': ['0-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-35', '36-40', '40+'],
            'data': [450, 680, 920, 1250, 1580, 1890, 1450, 1020, 760]
        },
        'recency': {
            'labels': ['0-30', '31-60', '61-90', '91-120', '121-150', '151-180', '180+'],
            'data': [2850, 2450, 1680, 1120, 890, 580, 430]
        }
    }
    
    # Build final API response
    api_data = {
        'metadata': {
            'projectName': 'Customer Segmentation System',
            'version': '1.0.0',
            'totalCustomers': len(data),
            'algorithm': pipeline.best_model.title() if pipeline.best_model else 'K-Means'
        },
        'kpi': {
            'totalCustomers': len(data),
            'totalRevenue': round(data['Total_Spending'].sum(), 2),
            'totalSegments': len(segments),
            'bestModel': pipeline.best_model.title() if pipeline.best_model else 'K-Means',
            'bestScore': round(best_results.get('silhouette', 0.32), 2),
            'avgSpending': round(data['Total_Spending'].mean(), 2),
            'avgPurchases': round(data['Num_Purchases'].mean(), 1),
            'churnRate': round((segments[0]['churnRisk'] == 'High' if segments else False) * 100, 1)
        },
        'segments': segments,
        'modelPerformance': model_performance,
        'insights': api_insights,
        'marketingStrategies': marketing_strategies,
        'distribution': distribution
    }
    
    return api_data


# ============================================
# API Routes
# ============================================

@app.route('/')
def index():
    """Serve the main dashboard."""
    return send_from_directory('frontend', 'index.html')


@app.route('/api/data')
def get_data():
    """
    Get all segmentation data for the dashboard.
    """
    try:
        data = prepare_api_data()
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/segments')
def get_segments():
    """Get customer segments."""
    try:
        data = prepare_api_data()
        return jsonify(data['segments'])
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/kpi')
def get_kpi():
    """Get KPI metrics."""
    try:
        data = prepare_api_data()
        return jsonify(data['kpi'])
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/insights')
def get_insights():
    """Get business insights."""
    try:
        data = prepare_api_data()
        return jsonify(data['insights'])
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/models')
def get_models():
    """Get model performance."""
    try:
        data = prepare_api_data()
        return jsonify(data['modelPerformance'])
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/marketing')
def get_marketing():
    """Get marketing strategies."""
    try:
        data = prepare_api_data()
        return jsonify(data['marketingStrategies'])
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/refresh')
def refresh_data():
    """
    Refresh data by re-running the pipeline.
    """
    global pipeline_results
    pipeline_results = None
    try:
        data = prepare_api_data()
        return jsonify({'status': 'success', 'message': 'Data refreshed successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Serve static files from frontend folder
@app.route('/<path:path>')
def serve_static(path):
    """Serve static files from frontend directory."""
    return send_from_directory('frontend', path)


# ============================================
# Main Entry Point
# ============================================

def main():
    """
    Run the Flask development server.
    """
    print("=" * 60)
    print("Customer Segmentation API Server")
    print("=" * 60)
    print("\nStarting pipeline...")
    
    # Pre-load data
    try:
        prepare_api_data()
        print("Pipeline completed successfully!\n")
    except Exception as e:
        print(f"Warning: Pipeline initialization error: {e}")
        print("API will run but may have limited data.\n")
    
    print("Starting Flask server...")
    print("Dashboard available at: http://localhost:5000")
    print("\nAPI Endpoints:")
    print("  - GET /api/data       - All data")
    print("  - GET /api/segments  - Customer segments")
    print("  - GET /api/kpi       - KPI metrics")
    print("  - GET /api/insights  - Business insights")
    print("  - GET /api/models    - Model performance")
    print("  - GET /api/marketing - Marketing strategies")
    print("  - GET /api/refresh   - Refresh data")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=True)


if __name__ == "__main__":
    main()

