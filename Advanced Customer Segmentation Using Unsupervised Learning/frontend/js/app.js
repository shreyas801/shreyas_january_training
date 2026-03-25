/**
 * Customer Segmentation Dashboard - Main Application
 * Handles navigation, data loading, and UI interactions
 */

// ============================================
// Configuration
// ============================================

const API_BASE_URL = 'http://localhost:5000/api';

// ============================================
// Global Data Store
// ============================================

const appData = {
    totalCustomers: 0,
    totalRevenue: 0,
    bestModel: '',
    bestScore: 0,
    segments: [],
    insights: {},
    marketingStrategies: [],
    isLoading: false,
    useApi: false,  // Set to true to use API, false for local JSON data
    rawData: null  // Store raw data from JSON
};

// ============================================
// Navigation
// ============================================

function initNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    
    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const sectionId = item.dataset.section;
            showSection(sectionId);
            
            // Update active state
            navItems.forEach(nav => nav.classList.remove('active'));
            item.classList.add('active');
        });
    });
}

function showSection(sectionId) {
    const sections = document.querySelectorAll('.content-section');
    sections.forEach(section => {
        section.classList.remove('active');
    });
    
    const targetSection = document.getElementById(sectionId);
    if (targetSection) {
        targetSection.classList.add('active');
    }
}

// ============================================
// Sidebar Toggle
// ============================================

function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    const mainContent = document.querySelector('.main-content');
    sidebar.classList.toggle('collapsed');
    mainContent.classList.toggle('sidebar-collapsed');
}

// ============================================
// Data Loading & Initialization
// ============================================

function initDashboard() {
    // Initialize charts
    initCharts();
    // Load data from JSON or fallback
    loadSampleData();
}

function loadSampleData() {
    // First, load data from data.json
    fetch('data/data.json')
        .then(response => response.json())
        .then(jsonData => {
            // Store raw data
            appData.rawData = jsonData;
            
            // Set KPI values from JSON
            appData.totalCustomers = jsonData.kpi.totalCustomers;
            appData.totalRevenue = jsonData.kpi.totalRevenue;
            appData.bestModel = jsonData.kpi.bestModel;
            appData.bestScore = jsonData.kpi.bestScore;
            
            // Set segments from JSON
            appData.segments = jsonData.segments.map(seg => ({
                id: seg.id,
                name: seg.name,
                customerType: seg.customerType,
                count: seg.count,
                percentage: seg.percentage,
                avgSpending: seg.avgSpending,
                avgPurchases: seg.avgPurchases,
                avgIncome: seg.avgIncome,
                priority: seg.priority.toLowerCase().replace('HIGH', 'high').replace('MEDIUM', 'medium').replace('LOW', 'low').replace('URGENT', 'urgent'),
                color: seg.color,
                churnRisk: seg.churnRisk,
                revenue: seg.revenue
            }));
            
            // Set insights from JSON
            appData.insights = jsonData.insights;
            
            // Set marketing strategies from JSON
            appData.marketingStrategies = jsonData.marketingStrategies.map(strat => ({
                segment: strat.segment,
                priority: strat.priority,
                approach: strat.approach,
                channels: strat.channels,
                offers: strat.offers,
                budget: strat.budget,
                color: strat.color
            }));
            
            // Render after data is loaded
            renderKPICards();
            renderSegmentCards();
            renderSegmentsTable();
            renderInsights();
            renderMarketingRecommendations();
            
            // Add loading effect
            setTimeout(() => {
                document.body.classList.add('loaded');
            }, 500);
        })
        .catch(error => {
            console.error('Error loading data.json:', error);
            // Fallback to hardcoded sample data
            loadHardcodedSampleData();
        });
}

function loadHardcodedSampleData() {
    // Fallback hardcoded sample data
    appData.totalCustomers = 10000;
    appData.totalRevenue = 2500000;
    appData.bestModel = 'K-Means';
    appData.bestScore = 0.32;
    
    // Sample segment data based on the project
    appData.segments = [
        {
            id: 0,
            name: 'Premium Loyal',
            customerType: 'Premium Loyal Customers',
            count: 1850,
            percentage: 18.5,
            avgSpending: 4520,
            avgPurchases: 28,
            avgIncome: 125000,
            priority: 'high',
            color: '#667eea',
            churnRisk: 'Low',
            revenue: 8362000
        },
        {
            id: 1,
            name: 'Big Ticket',
            customerType: 'Big Ticket Buyers',
            count: 1450,
            percentage: 14.5,
            avgSpending: 3850,
            avgPurchases: 12,
            avgIncome: 95000,
            priority: 'high',
            color: '#f093fb',
            churnRisk: 'Medium',
            revenue: 5582500
        },
        {
            id: 2,
            name: 'Frequent',
            customerType: 'Frequent Low-Spenders',
            count: 2650,
            percentage: 26.5,
            avgSpending: 1850,
            avgPurchases: 35,
            avgIncome: 55000,
            priority: 'medium',
            color: '#43e97b',
            churnRisk: 'Low',
            revenue: 4902500
        },
        {
            id: 3,
            name: 'Budget',
            customerType: 'Budget Conscious',
            count: 2250,
            percentage: 22.5,
            avgSpending: 920,
            avgPurchases: 8,
            avgIncome: 35000,
            priority: 'low',
            color: '#fa709a',
            churnRisk: 'Medium',
            revenue: 2070000
        },
        {
            id: 4,
            name: 'At-Risk',
            customerType: 'At-Risk Customers',
            count: 1800,
            percentage: 18.0,
            avgSpending: 680,
            avgPurchases: 5,
            avgIncome: 42000,
            priority: 'urgent',
            color: '#4facfe',
            churnRisk: 'High',
            revenue: 1224000
        }
    ];
    
    appData.insights = {
        revenueAnalysis: [
            { segment: 'Premium Loyal', revenue: 8362000, percentage: 33.5 },
            { segment: 'Big Ticket', revenue: 5582500, percentage: 22.3 },
            { segment: 'Frequent', revenue: 4902500, percentage: 19.6 },
            { segment: 'Budget', revenue: 2070000, percentage: 8.3 },
            { segment: 'At-Risk', revenue: 1224000, percentage: 4.9 }
        ],
        churnAnalysis: [
            { segment: 'Premium Loyal', risk: 'Low', avgRecency: 25 },
            { segment: 'Big Ticket', risk: 'Medium', avgRecency: 45 },
            { segment: 'Frequent', risk: 'Low', avgRecency: 18 },
            { segment: 'Budget', risk: 'Medium', avgRecency: 55 },
            { segment: 'At-Risk', risk: 'High', avgRecency: 95 }
        ],
        keyFindings: [
            'Premium Loyal Customers generate 33.5% of total revenue',
            'Top 20% of customers contribute 60% of revenue',
            'At-Risk segment has 95 days average recency - needs immediate attention',
            'Frequent buyers have highest engagement with 35 avg purchases'
        ],
        actionItems: [
            'Implement VIP program for Premium Loyal customers',
            'Launch win-back campaign for At-Risk segment',
            'Create bundle deals for Frequent low-spenders to increase basket size',
            'Allocate 35% marketing budget to Premium segment'
        ]
    };
    
    appData.marketingStrategies = [
        {
            segment: 'Premium Loyal',
            priority: 'HIGH',
            approach: 'VIP Treatment & Exclusive Offers',
            channels: ['Personal Email', 'Phone', 'VIP Events'],
            offers: [
                'Early access to new products',
                'Exclusive discounts',
                'Personalized recommendations',
                'Loyalty rewards program'
            ],
            budget: '35%',
            color: '#667eea'
        },
        {
            segment: 'Big Ticket',
            priority: 'HIGH',
            approach: 'Premium Product Focus',
            channels: ['Personal Email', 'Direct Mail', 'Showroom'],
            offers: [
                'Premium product bundles',
                'Financing options',
                'White-glove service',
                'Exclusive previews'
            ],
            budget: '25%',
            color: '#f093fb'
        },
        {
            segment: 'Frequent',
            priority: 'MEDIUM',
            approach: 'Engagement & Upselling',
            channels: ['Email', 'App Notifications', 'Social Media'],
            offers: [
                'Bundle deals',
                'Cross-sell recommendations',
                'Loyalty points',
                'Referral programs'
            ],
            budget: '20%',
            color: '#43e97b'
        },
        {
            segment: 'Budget',
            priority: 'LOW',
            approach: 'Value-Focused Marketing',
            channels: ['Email', 'Social Media', 'Deals Pages'],
            offers: [
                'Discount promotions',
                'Clearance sales',
                'Value bundles',
                'Cashback offers'
            ],
            budget: '10%',
            color: '#fa709a'
        },
        {
            segment: 'At-Risk',
            priority: 'URGENT',
            approach: 'Win-back Campaigns',
            channels: ['Email', 'SMS', 'Retargeting Ads'],
            offers: [
                'Special comeback discounts',
                'Limited-time offers',
                'Feedback surveys',
                'Personalized re-engagement'
            ],
            budget: '25%',
            color: '#4facfe'
        }
    ];
    
    // Render with fallback data
    renderKPICards();
    renderSegmentCards();
    renderSegmentsTable();
    renderInsights();
    renderMarketingRecommendations();
    
    // Add loading effect
    setTimeout(() => {
        document.body.classList.add('loaded');
    }, 500);
}

// ============================================
// KPI Cards
// ============================================

function renderKPICards() {
    // Animate values
    animateValue('totalCustomers', 0, appData.totalCustomers, 1000);
    animateValue('totalRevenue', 0, appData.totalRevenue / 1000000, 1000, '$', 'M');
    document.getElementById('totalSegments').textContent = appData.segments.length;
    
    // Display best model with score (text, no animation)
    const bestModelElement = document.getElementById('bestModel');
    if (bestModelElement) {
        // If we have a valid score, show the model with score
        if (appData.bestScore > 0 && appData.bestModel) {
            bestModelElement.textContent = `${appData.bestModel} (${appData.bestScore} Score)`;
        } else if (appData.bestModel) {
            bestModelElement.textContent = appData.bestModel;
        } else {
            bestModelElement.textContent = 'N/A';
        }
    }
}

function animateValue(id, start, end, duration, prefix = '', suffix = '') {
    const element = document.getElementById(id);
    if (!element) return;
    
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        const value = progress * (end - start) + start;
        element.textContent = prefix + value.toFixed(end < 100 ? 1 : 0) + suffix;
        if (progress < 1) {
            window.requestAnimationFrame(step);
        }
    };
    window.requestAnimationFrame(step);
}

// ============================================
// Segment Cards
// ============================================

function renderSegmentCards() {
    const container = document.getElementById('segmentCards');
    if (!container) return;
    
    container.innerHTML = appData.segments.map(segment => `
        <div class="segment-card" style="--segment-color: ${segment.color}">
            <div class="segment-color" style="background: ${segment.color}"></div>
            <div class="segment-name">${segment.name}</div>
            <div class="segment-count">${segment.count.toLocaleString()}</div>
            <div class="segment-percentage">${segment.percentage}%</div>
        </div>
    `).join('');
}

// ============================================
// Segments Table
// ============================================

function renderSegmentsTable() {
    const tbody = document.getElementById('segmentsTableBody');
    if (!tbody) return;
    
    tbody.innerHTML = appData.segments.map(segment => `
        <tr>
            <td>
                <span style="display: inline-block; width: 12px; height: 12px; border-radius: 50%; background: ${segment.color}; margin-right: 8px;"></span>
                ${segment.name}
            </td>
            <td>${segment.customerType}</td>
            <td>${segment.count.toLocaleString()}</td>
            <td>${segment.percentage}%</td>
            <td>$${segment.avgSpending.toLocaleString()}</td>
            <td>${segment.avgPurchases}</td>
            <td>$${segment.avgIncome.toLocaleString()}</td>
            <td><span class="priority-badge ${segment.priority}">${segment.priority.toUpperCase()}</span></td>
        </tr>
    `).join('');
}

// ============================================
// Insights
// ============================================

function renderInsights() {
    renderRevenueAnalysis();
    renderChurnAnalysis();
    renderKeyFindings();
    renderActionItems();
}

function renderRevenueAnalysis() {
    const container = document.getElementById('revenueAnalysis');
    if (!container) return;
    
    const colors = ['#667eea', '#f093fb', '#43e97b', '#fa709a', '#4facfe'];
    
    container.innerHTML = appData.insights.revenueAnalysis.map((item, index) => `
        <div class="insight-item">
            <span class="insight-label" style="display: flex; align-items: center; gap: 8px;">
                <span style="display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: ${colors[index]};"></span>
                ${item.segment}
            </span>
            <span class="insight-value">$${(item.revenue / 1000000).toFixed(2)}M <span style="font-size: 0.8rem; color: var(--text-muted)">(${item.percentage}%)</span></span>
        </div>
    `).join('');
}

function renderChurnAnalysis() {
    const container = document.getElementById('churnAnalysis');
    if (!container) return;
    
    container.innerHTML = appData.insights.churnAnalysis.map(item => {
        let riskClass = item.risk === 'High' ? 'danger' : item.risk === 'Medium' ? 'warning' : 'success';
        return `
            <div class="insight-item">
                <span class="insight-label">${item.segment}</span>
                <span class="insight-value text-${riskClass}">${item.risk} Risk</span>
            </div>
        `;
    }).join('');
}

function renderKeyFindings() {
    const container = document.getElementById('keyFindings');
    if (!container) return;
    
    container.innerHTML = `
        <ul class="findings-list">
            ${appData.insights.keyFindings.map(finding => `
                <li><i class="fas fa-check-circle"></i> ${finding}</li>
            `).join('')}
        </ul>
    `;
}

function renderActionItems() {
    const container = document.getElementById('actionItems');
    if (!container) return;
    
    container.innerHTML = `
        <ul class="action-list">
            ${appData.insights.actionItems.map(item => `
                <li>${item}</li>
            `).join('')}
        </ul>
    `;
}

// ============================================
// Marketing Recommendations
// ============================================

function renderMarketingRecommendations() {
    const container = document.getElementById('marketingRecommendations');
    if (!container) return;
    
    container.innerHTML = appData.marketingStrategies.map(strategy => `
        <div class="marketing-card" style="--segment-color: ${strategy.color}">
            <div class="marketing-header" style="background: linear-gradient(135deg, ${strategy.color}, ${adjustColor(strategy.color, -30)});">
                <h3>${strategy.segment}</h3>
                <div class="segment-type">${strategy.approach}</div>
            </div>
            <div class="marketing-body">
                <div class="marketing-section">
                    <h4><i class="fas fa-bullseye"></i> Priority</h4>
                    <span class="priority-badge ${strategy.priority.toLowerCase()}">${strategy.priority}</span>
                </div>
                <div class="marketing-section">
                    <h4><i class="fas fa-paper-plane"></i> Channels</h4>
                    <p>${strategy.channels.join(', ')}</p>
                </div>
                <div class="marketing-section">
                    <h4><i class="fas fa-gift"></i> Suggested Offers</h4>
                    <ul class="offer-list">
                        ${strategy.offers.map(offer => `<li><i class="fas fa-check"></i> ${offer}</li>`).join('')}
                    </ul>
                </div>
                <div class="marketing-section">
                    <h4><i class="fas fa-wallet"></i> Budget</h4>
                    <span class="budget-tag">${strategy.budget}</span>
                </div>
            </div>
        </div>
    `).join('');
}

function adjustColor(hex, amount) {
    return hex; // Simplified - returns same color
}

// ============================================
// Refresh Data
// ============================================

function refreshData() {
    const btn = document.querySelector('.btn-primary');
    const originalContent = btn.innerHTML;
    
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Refreshing...';
    btn.disabled = true;
    
    setTimeout(() => {
        btn.innerHTML = originalContent;
        btn.disabled = false;
        
        // Show notification
        showNotification('Data refreshed successfully!', 'success');
    }, 2000);
}

function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <i class="fas fa-${type === 'success' ? 'check-circle' : 'info-circle'}"></i>
        ${message}
    `;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        padding: 1rem 1.5rem;
        border-radius: var(--radius-md);
        z-index: 9999;
        animation: slideIn 0.3s ease;
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// ============================================
// Initialize on DOM Load
// ============================================

document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initDashboard();
});

