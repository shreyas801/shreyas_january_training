"""
Business Insights Generator Module
===================================
Generates business insights and recommendations for each customer segment:
- Revenue analysis
- Marketing strategy recommendations
- Retention strategies
- Personalized offer suggestions

Author: Student
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')


class BusinessInsightsGenerator:
    """
    A class to generate business insights and recommendations from clustering results.
    """
    
    def __init__(self, df, cluster_profiles):
        """
        Initialize business insights generator.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data with cluster labels
        cluster_profiles : dict
            Cluster profiles from ClusterAnalyzer
        """
        self.df = df.copy()
        self.cluster_profiles = cluster_profiles
        self.insights = {}
        
    def analyze_revenue_segments(self):
        """
        Analyze which segments generate the highest revenue.
        
        Returns:
        --------
        dict : Revenue analysis by segment
        """
        print("\n" + "=" * 60)
        print("REVENUE ANALYSIS BY SEGMENT")
        print("=" * 60)
        
        total_revenue = self.df['Total_Spending'].sum()
        
        revenue_analysis = {}
        
        for cluster_id in sorted(self.df['Cluster'].unique()):
            cluster_data = self.df[self.df['Cluster'] == cluster_id]
            
            cluster_revenue = cluster_data['Total_Spending'].sum()
            revenue_pct = (cluster_revenue / total_revenue) * 100
            
            avg_revenue = cluster_data['Total_Spending'].mean()
            customer_count = len(cluster_data)
            
            revenue_analysis[cluster_id] = {
                'total_revenue': cluster_revenue,
                'revenue_percentage': revenue_pct,
                'avg_revenue_per_customer': avg_revenue,
                'customer_count': customer_count,
                'revenue_per_customer': cluster_revenue / customer_count
            }
            
            print(f"\nCluster {cluster_id}:")
            print(f"  Total Revenue: ${cluster_revenue:,.2f}")
            print(f"  Revenue Share: {revenue_pct:.2f}%")
            print(f"  Avg Revenue/Customer: ${avg_revenue:,.2f}")
        
        # Identify top revenue generator
        top_cluster = max(revenue_analysis.items(), 
                        key=lambda x: x[1]['total_revenue'])[0]
        
        print(f"\n*** TOP REVENUE GENERATOR: Cluster {top_cluster} ***")
        
        return revenue_analysis
    
    def identify_churn_risk_segments(self):
        """
        Identify segments that are likely to churn.
        
        Returns:
        --------
        dict : Churn risk analysis by segment
        """
        print("\n" + "=" * 60)
        print("CHURN RISK ANALYSIS")
        print("=" * 60)
        
        overall_avg_recency = self.df['Recency'].mean()
        
        churn_analysis = {}
        
        for cluster_id in sorted(self.df['Cluster'].unique()):
            cluster_data = self.df[self.df['Cluster'] == cluster_id]
            
            avg_recency = cluster_data['Recency'].mean()
            recency_ratio = avg_recency / overall_avg_recency
            
            # Churn risk score based on recency
            if recency_ratio > 1.5:
                risk_level = "HIGH"
            elif recency_ratio > 1.2:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"
            
            churn_analysis[cluster_id] = {
                'avg_recency': avg_recency,
                'recency_ratio': recency_ratio,
                'risk_level': risk_level
            }
            
            print(f"\nCluster {cluster_id}:")
            print(f"  Average Recency: {avg_recency:.1f} days")
            print(f"  Recency Ratio: {recency_ratio:.2f}x average")
            print(f"  Churn Risk: {risk_level}")
        
        # Identify highest risk
        high_risk_clusters = [
            cid for cid, info in churn_analysis.items() 
            if info['risk_level'] == 'HIGH'
        ]
        
        if high_risk_clusters:
            print(f"\n*** HIGH CHURN RISK SEGMENTS: Clusters {high_risk_clusters} ***")
        
        return churn_analysis
    
    def generate_marketing_recommendations(self):
        """
        Generate marketing strategy recommendations for each segment.
        
        Returns:
        --------
        dict : Marketing recommendations by segment
        """
        print("\n" + "=" * 60)
        print("MARKETING RECOMMENDATIONS")
        print("=" * 60)
        
        marketing_strategies = {}
        
        for cluster_id, profile in self.cluster_profiles.items():
            customer_type = profile.get('customer_type', 'Average Customers')
            
            # Get cluster-specific metrics
            cluster_data = self.df[self.df['Cluster'] == cluster_id]
            avg_spending = cluster_data['Total_Spending'].mean()
            avg_purchases = cluster_data['Num_Purchases'].mean()
            avg_income = cluster_data['Income'].mean()
            
            # Generate recommendations based on customer type
            if 'Premium' in customer_type or 'High-Value' in customer_type:
                strategy = {
                    'segment': customer_type,
                    'priority': 'HIGH',
                    'marketing_approach': 'VIP Treatment & Exclusive Offers',
                    'channels': ['Personal Email', 'Phone', 'VIP Events'],
                    'offers': [
                        'Early access to new products',
                        'Exclusive discounts',
                        'Personalized recommendations',
                        'Loyalty rewards program'
                    ],
                    'budget_allocation': '35%'
                }
            elif 'At-Risk' in customer_type or 'Risk' in customer_type:
                strategy = {
                    'segment': customer_type,
                    'priority': 'URGENT',
                    'marketing_approach': 'Win-back Campaigns',
                    'channels': ['Email', 'SMS', 'Retargeting Ads'],
                    'offers': [
                        'Special comeback discounts',
                        'Limited-time offers',
                        'Feedback surveys',
                        'Personalized re-engagement emails'
                    ],
                    'budget_allocation': '25%'
                }
            elif 'Frequent' in customer_type:
                strategy = {
                    'segment': customer_type,
                    'priority': 'MEDIUM',
                    'marketing_approach': 'Engagement & Upselling',
                    'channels': ['Email', 'App Notifications', 'Social Media'],
                    'offers': [
                        'Bundle deals',
                        'Cross-sell recommendations',
                        'Loyalty points',
                        'Referral programs'
                    ],
                    'budget_allocation': '20%'
                }
            elif 'Budget' in customer_type or 'Low' in customer_type:
                strategy = {
                    'segment': customer_type,
                    'priority': 'LOW',
                    'marketing_approach': 'Value-Focused Marketing',
                    'channels': ['Email', 'Social Media', 'Deals Pages'],
                    'offers': [
                        'Discount promotions',
                        'Clearance sales',
                        'Value bundles',
                        'Cashback offers'
                    ],
                    'budget_allocation': '10%'
                }
            else:  # Average customers
                strategy = {
                    'segment': customer_type,
                    'priority': 'MEDIUM',
                    'marketing_approach': 'Nurture & Convert',
                    'channels': ['Email', 'Social Media', 'Website'],
                    'offers': [
                        'Welcome series',
                        'Educational content',
                        'Seasonal promotions',
                        'Product recommendations'
                    ],
                    'budget_allocation': '10%'
                }
            
            marketing_strategies[cluster_id] = strategy
            
            print(f"\n{'='*50}")
            print(f"CLUSTER {cluster_id}: {customer_type}")
            print(f"{'='*50}")
            print(f"Priority: {strategy['priority']}")
            print(f"Marketing Approach: {strategy['marketing_approach']}")
            print(f"Recommended Channels: {', '.join(strategy['channels'])}")
            print(f"Suggested Offers:")
            for offer in strategy['offers']:
                print(f"  - {offer}")
            print(f"Budget Allocation: {strategy['budget_allocation']}")
        
        return marketing_strategies
    
    def generate_retention_strategies(self):
        """
        Generate retention strategies for each segment.
        
        Returns:
        --------
        dict : Retention strategies by segment
        """
        print("\n" + "=" * 60)
        print("RETENTION STRATEGIES")
        print("=" * 60)
        
        retention_strategies = {}
        
        for cluster_id, profile in self.cluster_profiles.items():
            customer_type = profile.get('customer_type', 'Average Customers')
            
            cluster_data = self.df[self.df['Cluster'] == cluster_id]
            avg_tenure = cluster_data['Tenure_Months'].mean() if 'Tenure_Months' in cluster_data.columns else 12
            
            # Generate retention strategies
            if 'Premium' in customer_type or 'High-Value' in customer_type:
                strategy = {
                    'focus': 'Maintain Engagement',
                    'tactics': [
                        'Dedicated account manager',
                        'White-glove service',
                        'Exclusive product launches',
                        'Annual VIP appreciation events',
                        'Customized shopping experiences'
                    ],
                    'kpi_to_monitor': ['NPS', 'Repeat Purchase Rate', 'Customer Lifetime Value']
                }
            elif 'At-Risk' in customer_type or 'Risk' in customer_type:
                strategy = {
                    'focus': 'Re-engagement & Recovery',
                    'tactics': [
                        'Win-back email campaigns',
                        'Special comeback offers',
                        'Customer satisfaction surveys',
                        'Personalized outreach',
                        'Address pain points directly'
                    ],
                    'kpi_to_monitor': ['Recovery Rate', 'Churn Rate', 'Engagement Score']
                }
            elif 'Frequent' in customer_type:
                strategy = {
                    'focus': 'Increase Loyalty',
                    'tactics': [
                        'Loyalty rewards program',
                        'Points accumulation system',
                        'Exclusive member benefits',
                        'Personalized recommendations',
                        'Referral incentives'
                    ],
                    'kpi_to_monitor': ['Loyalty Program Enrollment', 'Referral Rate', 'Basket Size']
                }
            else:
                strategy = {
                    'focus': 'Build Relationship',
                    'tactics': [
                        'Regular communication',
                        'Educational content',
                        'Community building',
                        'Feedback collection',
                        'Gradual engagement increase'
                    ],
                    'kpi_to_monitor': ['Engagement Rate', 'Email Open Rate', 'Conversion Rate']
                }
            
            retention_strategies[cluster_id] = strategy
            
            print(f"\nCluster {cluster_id} ({customer_type}):")
            print(f"  Focus: {strategy['focus']}")
            print(f"  Key Tactics:")
            for tactic in strategy['tactics']:
                print(f"    - {tactic}")
        
        return retention_strategies
    
    def generate_offer_suggestions(self):
        """
        Generate personalized offer suggestions for each segment.
        
        Returns:
        --------
        dict : Offer suggestions by segment
        """
        print("\n" + "=" * 60)
        print("PERSONALIZED OFFER SUGGESTIONS")
        print("=" * 60)
        
        offer_suggestions = {}
        
        for cluster_id, profile in self.cluster_profiles.items():
            customer_type = profile.get('customer_type', 'Average Customers')
            
            cluster_data = self.df[self.df['Cluster'] == cluster_id]
            avg_spending = cluster_data['Total_Spending'].mean()
            spending_score = cluster_data['Spending_Score'].mean()
            
            # Generate offers based on spending behavior
            if spending_score > 70:
                offers = [
                    'Premium membership with exclusive benefits',
                    'Luxury brand collaborations',
                    'Personal shopping assistant',
                    'VIP-only collections'
                ]
            elif spending_score > 50:
                offers = [
                    'Mid-tier loyalty program',
                    'Seasonal bundles at 20% off',
                    'Free shipping on orders over $50',
                    'Birthday special discounts'
                ]
            elif spending_score > 30:
                offers = [
                    'First-time buyer discounts',
                    'Buy-one-get-one offers',
                    'Clearance section access',
                    'Student/military discounts'
                ]
            else:
                offers = [
                    'Deep discount promotions',
                    'Value packs',
                    'Mail-in rebates',
                    'Budget-friendly bundles'
                ]
            
            offer_suggestions[cluster_id] = {
                'customer_type': customer_type,
                'avg_spending_score': spending_score,
                'recommended_offers': offers
            }
            
            print(f"\nCluster {cluster_id} - {customer_type}:")
            print(f"  Avg Spending Score: {spending_score:.1f}")
            print(f"  Recommended Offers:")
            for offer in offers:
                print(f"    - {offer}")
        
        return offer_suggestions
    
    def create_business_summary(self, revenue_analysis, churn_analysis, 
                                marketing_strategies, retention_strategies):
        """
        Create a comprehensive business summary.
        
        Returns:
        --------
        dict : Business summary
        """
        print("\n" + "=" * 60)
        print("EXECUTIVE BUSINESS SUMMARY")
        print("=" * 60)
        
        # Key findings
        total_customers = len(self.df)
        total_revenue = self.df['Total_Spending'].sum()
        
        summary = {
            'total_customers': total_customers,
            'total_revenue': total_revenue,
            'n_segments': len(self.cluster_profiles),
            'key_findings': [],
            'action_items': []
        }
        
        # Find highest value segment
        highest_value = max(
            self.cluster_profiles.items(),
            key=lambda x: x[1].get('avg_total_spending', 0)
        )
        
        summary['key_findings'].append(
            f"Highest Value Segment: Cluster {highest_value[0]} "
            f"({highest_value[1].get('customer_type', 'N/A')}) "
            f"with avg spending ${highest_value[1].get('avg_total_spending', 0):,.2f}"
        )
        
        # Find highest frequency segment
        highest_freq = max(
            self.cluster_profiles.items(),
            key=lambda x: x[1].get('avg_num_purchases', 0)
        )
        
        summary['key_findings'].append(
            f"Most Frequent Buyers: Cluster {highest_freq[0]} "
            f"({highest_freq[1].get('customer_type', 'N/A')}) "
            f"with {highest_freq[1].get('avg_num_purchases', 0):.1f} avg purchases"
        )
        
        # Find at-risk segments
        at_risk_clusters = [
            cid for cid, info in churn_analysis.items()
            if info.get('risk_level') == 'HIGH'
        ]
        
        if at_risk_clusters:
            summary['key_findings'].append(
                f"At-Risk Segments: Clusters {at_risk_clusters} require immediate attention"
            )
        
        # Top action items
        summary['action_items'] = [
            "Implement personalized marketing campaigns for each segment",
            "Develop retention programs for high-risk customers",
            "Create VIP programs for premium customer segments",
            "Optimize budget allocation based on segment value",
            "Monitor key metrics per segment weekly"
        ]
        
        print(f"\nTotal Customers: {total_customers:,}")
        print(f"Total Revenue: ${total_revenue:,.2f}")
        print(f"Number of Segments: {len(self.cluster_profiles)}")
        
        print("\nKey Findings:")
        for finding in summary['key_findings']:
            print(f"  • {finding}")
        
        print("\nPriority Action Items:")
        for i, action in enumerate(summary['action_items'], 1):
            print(f"  {i}. {action}")
        
        return summary
    
    def generate_insights(self):
        """
        Execute complete insight generation pipeline.
        
        Returns:
        --------
        dict : Complete business insights
        """
        print("\n" + "=" * 60)
        print("BUSINESS INSIGHTS GENERATION")
        print("=" * 60)
        
        # 1. Revenue analysis
        revenue_analysis = self.analyze_revenue_segments()
        
        # 2. Churn risk analysis
        churn_analysis = self.identify_churn_risk_segments()
        
        # 3. Marketing recommendations
        marketing_strategies = self.generate_marketing_recommendations()
        
        # 4. Retention strategies
        retention_strategies = self.generate_retention_strategies()
        
        # 5. Offer suggestions
        offer_suggestions = self.generate_offer_suggestions()
        
        # 6. Executive summary
        summary = self.create_business_summary(
            revenue_analysis, churn_analysis,
            marketing_strategies, retention_strategies
        )
        
        # Compile all insights
        self.insights = {
            'revenue_analysis': revenue_analysis,
            'churn_analysis': churn_analysis,
            'marketing_strategies': marketing_strategies,
            'retention_strategies': retention_strategies,
            'offer_suggestions': offer_suggestions,
            'summary': summary
        }
        
        print("\n" + "=" * 60)
        print("INSIGHTS GENERATION COMPLETE")
        print("=" * 60)
        
        return self.insights
    
    def print_insights(self, insights):
        """
        Print formatted insights.
        
        Parameters:
        -----------
        insights : dict
            Business insights
        """
        print("\n" + "=" * 80)
        print("DETAILED BUSINESS INSIGHTS")
        print("=" * 80)
        
        # Print summary
        print("\n### EXECUTIVE SUMMARY ###")
        summary = insights['summary']
        print(f"Total Customers: {summary['total_customers']:,}")
        print(f"Total Revenue: ${summary['total_revenue']:,.2f}")
        print(f"Segments Identified: {summary['n_segments']}")
        
        print("\n### KEY FINDINGS ###")
        for finding in summary['key_findings']:
            print(f"  • {finding}")
        
        print("\n### ACTION ITEMS ###")
        for i, action in enumerate(summary['action_items'], 1):
            print(f"  {i}. {action}")
    
    def save_insights_report(self, insights):
        """
        Save insights to a report file.
        
        Parameters:
        -----------
        insights : dict
            Business insights
        """
        print("\n--- Saving Insights Report ---")
        
        # Create a comprehensive report
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("CUSTOMER SEGMENTATION - BUSINESS INSIGHTS REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Executive Summary
        summary = insights['summary']
        report_lines.append("### EXECUTIVE SUMMARY ###")
        report_lines.append(f"Total Customers: {summary['total_customers']:,}")
        report_lines.append(f"Total Revenue: ${summary['total_revenue']:,.2f}")
        report_lines.append(f"Segments Identified: {summary['n_segments']}")
        report_lines.append("")
        
        # Key Findings
        report_lines.append("### KEY FINDINGS ###")
        for finding in summary['key_findings']:
            report_lines.append(f"• {finding}")
        report_lines.append("")
        
        # Action Items
        report_lines.append("### PRIORITY ACTION ITEMS ###")
        for i, action in enumerate(summary['action_items'], 1):
            report_lines.append(f"{i}. {action}")
        report_lines.append("")
        
        # Segment Details
        report_lines.append("### SEGMENT DETAILS ###")
        for cluster_id, profile in self.cluster_profiles.items():
            report_lines.append(f"\n--- Cluster {cluster_id} ---")
            report_lines.append(f"Customer Type: {profile.get('customer_type', 'N/A')}")
            report_lines.append(f"Size: {profile.get('size', 0)} customers ({profile.get('percentage', 0):.1f}%)")
            report_lines.append(f"Average Spending: ${profile.get('avg_total_spending', 0):,.2f}")
            report_lines.append(f"Average Purchases: {profile.get('avg_num_purchases', 0):.1f}")
            report_lines.append(f"Average Income: ${profile.get('avg_income', 0):,.2f}")
        
        # Write to file
        with open('reports/business_insights.txt', 'w') as f:
            f.write('\n'.join(report_lines))
        
        # Save as CSV
        profiles_df = pd.DataFrame(self.cluster_profiles).T
        profiles_df.to_csv('reports/segment_profiles.csv')
        
        print("Insights report saved to: reports/business_insights.txt")
        print("Segment profiles saved to: reports/segment_profiles.csv")

